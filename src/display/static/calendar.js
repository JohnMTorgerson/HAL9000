(() => {
    const qs = new URLSearchParams(location.search);

    // -------- options from query ----------
    const raw = qs.get("events") || qs.get("items") || qs.get("data");
    const title = qs.get("title") || "Schedule";
    const code = qs.get("code") || "GPM 72–KC";
    const accent = qs.get("accent");   // e.g. #FF5A5A
    const limit = Number(qs.get("limit") || 0); // 0 = unlimited

    console.log("events data:", raw);

    // -------- tiny utils ----------
    function tryParse(s) {
        if (!s) return null;
        try { return JSON.parse(s); } catch (_) { }
        try { return JSON.parse(decodeURIComponent(s)); } catch (_) { }
        try { return JSON.parse(atob(s)); } catch (_) { }
        return null;
    }
    function pad(n) { return n < 10 ? "0" + n : "" + n; }

    function parseDateMaybe(v) {
        if (!v) return null;
        if (v instanceof Date) return v;
        const t = (typeof v === "number") ? v : Date.parse(v);
        return isNaN(t) ? null : new Date(t);
    }

    // --- NEW: compact, non-wrapping date/time using epoch ts when available ---
    function _ms(ts) {
        if (ts == null) return null;
        ts = Number(ts);
        return ts < 1e12 ? ts * 1000 : ts; // seconds -> ms
    }
    const _dFmt = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" });
    function _hm(dt) {
        // "11:00 AM" -> "11am", "3:30 PM" -> "3:30pm"
        return dt.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true })
            .toLowerCase().replace(":00", "").replace(" ", "");
    }
    function formatAbbrevRangeTS(start_ts, end_ts) {
        if (start_ts == null) return null;
        const sMS = _ms(start_ts);
        if (!Number.isFinite(sMS)) return null;

        const s = new Date(sMS);
        const hasEnd = end_ts != null && Number.isFinite(_ms(end_ts));
        const e = hasEnd ? new Date(_ms(end_ts)) : null;

        let out = `${_dFmt.format(s)} ${_hm(s)}`;
        if (e) {
            out += (s.toDateString() === e.toDateString())
                ? `–${_hm(e)}`
                : ` – ${_dFmt.format(e)} ${_hm(e)}`;
        }
        return out;
    }

    // Fallback formatter (uses strings or numbers if no ts provided)
    function fmtRange(s, e) {
        const sd = parseDateMaybe(s), ed = parseDateMaybe(e);
        if (sd && ed) {
            const sameDay = sd.toDateString() === ed.toDateString();
            const dayStr = _dFmt.format(sd);
            const sStr = _hm(sd);
            const eStr = _hm(ed);
            return sameDay ? `${dayStr} ${sStr}–${eStr}` : `${dayStr} ${sStr} – ${_dFmt.format(ed)} ${eStr}`;
        }
        if (sd && !e) return `${_dFmt.format(sd)} ${_hm(sd)}`;
        if (typeof s === "string" && typeof e === "string") return `${s} – ${e}`;
        return s || "";
    }

    function isNowWindow(start_ts, end_ts, s, e) {
        // Prefer timestamps if present, else fall back to parsed dates
        const sMS = _ms(start_ts);
        const eMS = _ms(end_ts);
        if (Number.isFinite(sMS) && Number.isFinite(eMS)) {
            const now = Date.now();
            return now >= sMS && now <= eMS;
        }
        const sd = parseDateMaybe(s), ed = parseDateMaybe(e);
        if (!sd || !ed) return false;
        const now = Date.now();
        return now >= sd.getTime() && now <= ed.getTime();
    }

    // -------- render ----------
    const tryParsed = tryParse(raw);
    const data = Array.isArray(tryParsed) ? tryParsed : (tryParsed && tryParsed.hasOwnProperty("title")) ? [tryParsed] : [];
    console.log("parsed events:", data);

    const panel = document.getElementById("panel");
    const list = document.getElementById("events");

    // accent override
    if (accent) {
        panel.style.setProperty("--accent", accent);
        panel.style.setProperty("--row-border", `${accent}33`);
    }

    // header text
    document.getElementById("title").textContent = title.toUpperCase();
    document.getElementById("code").textContent = code;

    // normalize / sort by start (prefer timestamps for reliability)
    const items = data.map(ev => ({
        title: ev.title || ev.name || "(Untitled)",
        location: ev.location || "",
        start: ev.start || ev.start_dt || ev.startDate,
        end: ev.end || ev.end_dt || ev.endDate,
        start_ts: ev.start_ts ?? ev.startTS ?? null,
        end_ts: ev.end_ts ?? ev.endTS ?? null,
        open_now: ev.open_now,
    }));

    items.sort((a, b) => {
        const A = Number.isFinite(_ms(a.start_ts)) ? _ms(a.start_ts) : (parseDateMaybe(a.start)?.getTime() ?? Number.MAX_SAFE_INTEGER);
        const B = Number.isFinite(_ms(b.start_ts)) ? _ms(b.start_ts) : (parseDateMaybe(b.start)?.getTime() ?? Number.MAX_SAFE_INTEGER);
        return A - B;
    });

    const shown = limit > 0 ? items.slice(0, limit) : items;

    list.innerHTML = shown.map(ev => {
        const compact = formatAbbrevRangeTS(ev.start_ts, ev.end_ts) || fmtRange(ev.start, ev.end);
        const nowCls = isNowWindow(ev.start_ts, ev.end_ts, ev.start, ev.end) ? " now" : "";
        const tag = ev.open_now === true ? `<span class="tag">OPEN</span>` :
            ev.open_now === false ? `<span class="tag">CLOSED</span>` : "";
        const loc = ev.location ? `<div class="loc">${ev.location}</div>` : "";

        // time on its own line, title below (indented by CSS grid)
        return `
      <div class="event${nowCls}">
        <div class="time">${compact}</div>
        <div class="summary">
          <div class="title">${ev.title}${tag}</div>
          ${loc}
        </div>
      </div>
    `;
    }).join("");
})();
