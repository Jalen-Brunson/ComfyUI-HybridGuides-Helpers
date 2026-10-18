import re
_TAG_RES = {k: re.compile(r"<" + k + r" (\d+)>") for k in ("Picture", "Video", "Audio")}

def apply_tag_remap(text, pairs):
    """Simultaneous tag substitution (safe for swaps)."""
    placeholders = {}
    for i, (old, new) in enumerate(pairs):
        ph = f"\x00REMAP{i}\x00"
        placeholders[ph] = new
        text = text.replace(old, ph)
    for ph, new in placeholders.items():
        text = text.replace(ph, new)
    return text

def find_tag_numbers(text, kind):
    """All <kind N> numbers referenced in `text` -> set of ints."""
    return {int(m) for m in _TAG_RES[kind].findall(text or "")}

def compact_tag_pairs(kind, active_slots):
    """Active declared slot numbers -> (remap pairs excluding identities,
    {declared: compact}). E.g. [2, 3, 9] -> 2->1, 3->2, 9->3."""
    mapping = {slot: i + 1 for i, slot in enumerate(sorted(active_slots))}
    pairs = [(f"<{kind} {a}>", f"<{kind} {c}>")
             for a, c in mapping.items() if a != c]
    return pairs, mapping

class HGTagRenumber:
    """Queue-time renumber/validate for H3 ref tags.

    Prompt files reference stable DECLARED slot numbers (<Picture 9> = the
    sampler's slot-9 loader, forever). Core's MiniMaxH3ReferenceToVideo skips
    gated-off refs and the text encoder labels the survivors with a compact
    counter — this node sits between PromptChunkByIndex and the H3 ref node,
    passes every ref through unaltered, and rewrites the prompt's tags from
    declared numbers to the compact numbering the wire actually presents.
    A prompt referencing a gated-off slot errors (strict) instead of silently
    binding to the wrong picture.
    """

    N_PICS, N_VIDS, N_AUDS = 9, 3, 2

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, cls.N_PICS + 1):
            opt[f"pic_{i}"] = ("IMAGE", {"tooltip": f"Passthrough for declared <Picture {i}> (wire loader→RMBG→HGImageGate→here→ref_image_{i - 1})."})
        for i in range(1, cls.N_VIDS + 1):
            opt[f"video_{i}"] = ("IMAGE", {"tooltip": f"Passthrough for declared <Video {i}> (control video frames)."})
        for i in range(1, cls.N_AUDS + 1):
            opt[f"audio_{i}"] = ("AUDIO", {"tooltip": f"Passthrough for declared <Audio {i}>."})
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "strict": ("BOOLEAN", {"default": True, "tooltip": "Error when the prompt references a slot that is gated off / unwired. Off: warn and leave the tag as written."}),
                "renumber_videos": ("BOOLEAN", {"default": True, "tooltip": "Also renumber <Video N>/<Audio N> from the video_/audio_ passthroughs. Turn OFF for legacy 'loop tail: video_ref' prompts where <Video 1>/<Audio 1> mean the carried tail, not an input here."}),
                "unreferenced": (["warn", "drop", "error"], {"default": "warn", "tooltip": "A slot that is gated ON / wired but never referenced by the prompt's tags. Such a reference is STILL attended by the model on every step (the tag only tells the text encoder which picture is which) -- this is how an accidentally open Pic gate or a stale Control C leaks into a render. warn = pass it through with a console note (legacy); drop = withhold it (None downstream, the rest renumbered as if it were gated off); error = fail the queue."}),
            },
            "optional": opt,
            "hidden": {"api_prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (("STRING", "STRING")
                    + ("IMAGE",) * N_PICS + ("IMAGE",) * N_VIDS + ("AUDIO",) * N_AUDS
                    + ("STRING",))
    RETURN_NAMES = (("prompt", "mapping")
                    + tuple(f"pic_{i}" for i in range(1, N_PICS + 1))
                    + tuple(f"video_{i}" for i in range(1, N_VIDS + 1))
                    + tuple(f"audio_{i}" for i in range(1, N_AUDS + 1))
                    + ("index_report",))
    FUNCTION = "renumber"
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = "Rewrite <Picture/Video/Audio N> tags from stable declared slot numbers to the compact active numbering the H3 ref node actually presents, validating that every referenced slot is wired and gated on. Wire refs THROUGH this node into MiniMaxH3ReferenceToVideo."

    def _index_report(self, slots, api_prompt, unique_id):
        """Human-readable map of the cond_video_latents order this node feeds:
        ref images first (wire order), video controls LAST — so negative
        indices (MiniMaxH3ControlStrengthSchedule control_index) count back
        from the last control. Upstream node titles are resolved from the
        queued graph so each entry shows WHICH loader/gate it is."""
        def src_label(input_name):
            try:
                link = api_prompt[str(unique_id)]["inputs"].get(input_name)
                if not isinstance(link, list):
                    return ""
                src = api_prompt.get(str(link[0])) or {}
                title = (src.get("_meta") or {}).get("title") or src.get("class_type", "?")
                if "Gate" in src.get("class_type", ""):
                    inner = src.get("inputs", {}).get("image") or src.get("inputs", {}).get("audio")
                    if isinstance(inner, list):
                        s2 = api_prompt.get(str(inner[0])) or {}
                        t2 = (s2.get("_meta") or {}).get("title") or s2.get("class_type", "")
                        if t2:
                            title = f"{title} <- {t2}"
                return title
            except Exception:
                return ""

        pics = [i for i in range(1, self.N_PICS + 1) if slots.get(f"pic_{i}") is not None]
        vids = [i for i in range(1, self.N_VIDS + 1) if slots.get(f"video_{i}") is not None]
        auds = [i for i in range(1, self.N_AUDS + 1) if slots.get(f"audio_{i}") is not None]
        total = len(pics) + len(vids)
        lines = ["=== H3 cond index map (refs first, controls LAST) ==="]
        idx = 0
        lines.append(f"ref images ({len(pics)} active of {self.N_PICS}):")
        for k, i in enumerate(pics, 1):
            lab = src_label(f"pic_{i}")
            lines.append(f"  cond[{idx}]  ({idx - total:+d})  <Picture {k}>  pic_{i}"
                         + (f"  = {lab}" if lab else ""))
            idx += 1
        lines.append(f"control videos ({len(vids)} active of {self.N_VIDS}):")
        last_label = None
        for k, i in enumerate(vids, 1):
            lab = src_label(f"video_{i}")
            lines.append(f"  cond[{idx}]  ({idx - total:+d})  <Video {k}>  video_{i}"
                         + (f"  = {lab}" if lab else ""))
            if idx == total - 1:
                last_label = lab or f"video_{i}"
            idx += 1
        for i in auds:
            lab = src_label(f"audio_{i}")
            lines.append(f"audio ref: audio_{i}" + (f"  = {lab}" if lab else "")
                         + "  (not in cond_video_latents)")
        if total:
            tail = last_label if last_label else (
                f"pic_{pics[-1]} (NO control videos active!)" if pics else "?")
            lines.append(f">>> control_index -1 = {tail}")
        lines.append("(negatives are phase-safe; an AddGuide loop-carry keyframe"
                     " shifts POSITIVE indices +1, negatives unchanged)")
        return "\n".join(lines)

    def renumber(self, prompt, strict, renumber_videos, unreferenced="warn", **slots):
        api_prompt = slots.pop("api_prompt", None)
        unique_id = slots.pop("unique_id", None)
        kinds = [("Picture", "pic_", self.N_PICS)]
        if renumber_videos:
            kinds.append(("Video", "video_", self.N_VIDS))
            if any(slots.get(f"audio_{i}") is not None
                   for i in range(1, self.N_AUDS + 1)):
                kinds.append(("Audio", "audio_", self.N_AUDS))

        all_pairs, map_bits, problems, hints, dropped = [], [], [], [], []
        for kind, prefix, count in kinds:
            active = [i for i in range(1, count + 1)
                      if slots.get(f"{prefix}{i}") is not None]
            pairs, mapping = compact_tag_pairs(kind, active)
            referenced = find_tag_numbers(prompt, kind)
            bad = sorted(referenced - set(active))
            if bad:
                problems.append(
                    f"<{kind} {'/'.join(map(str, bad))}> referenced but "
                    f"slot(s) inactive (active {kind.lower()}s: "
                    f"{', '.join(map(str, active)) or 'none'})")
            unused = sorted(set(active) - referenced)
            if unused and unreferenced == "error":
                raise ValueError(
                    f"HGTagRenumber: {kind.lower()} slot(s) {', '.join(map(str, unused))} "
                    "active (gate on / wired) but never referenced in this prompt. An "
                    "unreferenced reference STILL conditions the model on every step -- "
                    "gate it off, or bind it in the prep flow so the prompt names it "
                    "(unreferenced=error).")
            if unused and unreferenced == "drop":
                # withhold the stray refs: None downstream, numbering as if gated off
                for i in unused:
                    slots[f"{prefix}{i}"] = None
                dropped.append(f"{kind.lower()} {', '.join(map(str, unused))}")
                active = [i for i in active if i not in unused]
                pairs, mapping = compact_tag_pairs(kind, active)
                unused = []
            if unused:
                hints.append(f"{kind.lower()} slot(s) "
                             f"{', '.join(map(str, unused))} active but never "
                             "referenced in this chunk -- it STILL conditions the model "
                             "(unreferenced=drop withholds it)")
            all_pairs += pairs
            map_bits.append(f"{kind[0]}: " + (", ".join(
                f"{d}→{c}" for d, c in mapping.items()) or "none active"))

        mapping_str = " | ".join(map_bits)
        if problems:
            msg = ("HGTagRenumber: " + "; ".join(problems)
                   + ". Enable the gate / wire the slot, or fix the prompt's "
                   "declared tags. Mapping: " + mapping_str)
            if strict:
                raise ValueError(msg)
            print(f"[HGTagRenumber] WARNING: {msg}")
        for hint in hints:
            print(f"[HGTagRenumber] note: {hint}")
        if dropped:
            print("[HGTagRenumber] DROPPED unreferenced slot(s), not sent to the model: "
                  + "; ".join(dropped))

        new_prompt = apply_tag_remap(prompt, all_pairs) if all_pairs else prompt
        if all_pairs:
            print(f"[HGTagRenumber] {mapping_str}")
        outs = [slots.get(f"pic_{i}") for i in range(1, self.N_PICS + 1)]
        outs += [slots.get(f"video_{i}") for i in range(1, self.N_VIDS + 1)]
        outs += [slots.get(f"audio_{i}") for i in range(1, self.N_AUDS + 1)]
        try:
            index_report = self._index_report(slots, api_prompt, unique_id)
        except Exception as e:  # the report is informational; never fail the run
            index_report = f"(index report unavailable: {e})"
        return (new_prompt, mapping_str, *outs, index_report)
