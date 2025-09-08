// static/app.js

(() => {
    const panels = {
        top: document.getElementById("top"),
        bottom: document.getElementById("bottom"),
    };

    const badge = document.getElementById("badge"); // optional status badge

    let ws;
    let prev = null; // previous render payload (deep-frozen snapshot)

    // ---- Utilities ----
    const deepClone = (o) => JSON.parse(JSON.stringify(o));
    const sameLayout = (a, b) => a === b;

    const samePanel = (a, b) => {
        if (!a || !b) return false;
        // Compare structural fields
        if (a.type !== b.type) return false;
        if (a.fit !== b.fit) return false;
        if ((a.bg || "#000") !== (b.bg || "#000")) return false;

        // For images/URLs, compare the "src"
        if ((a.type === "image" || a.type === "url") && a.src !== b.src) return false;

        // For text, treat structure as "same" even if the text changed
        // (we'll update text content in-place without animation)
        return true;
    };

    const setBadge = (text, cls) => {
        if (!badge) return;
        badge.textContent = text;
        badge.className = "badge " + (cls || "");
        badge.style.display = "block";
        clearTimeout(setBadge._t);
        setBadge._t = setTimeout(() => (badge.style.display = "none"), 1200);
    };

    // Apply / remove fullscreen layout class
    function applyLayout(mode) {
        const isFull = mode === "fullscreen";
        if (isFull) {
            document.body.classList.add("layout-fullscreen");
            panels.bottom.style.display = "none";
        } else {
            document.body.classList.remove("layout-fullscreen");
            panels.bottom.style.display = "";
        }
    }

    // Create a content element for a Panel
    function createContentEl(panel) {
        let el;
        if (panel.type === "image") {
            el = document.createElement("img");
            el.src = panel.src || "";
            el.className = panel.fit === "contain" ? "contain" : "cover";
        } else if (panel.type === "url") {
            el = document.createElement("iframe");
            el.src = panel.src || "about:blank";
        } else {
            el = document.createElement("div");
            el.className = "text";
            el.textContent = panel.text || "";
        }
        // Restrict fade animation to the content element only (never the container)
        el.classList.add("fade");
        return el;
    }

    // Render/update a single slot
    function renderSlot(slot, nextPanel, prevPanel) {
        const container = panels[slot];
        if (!container) return;

        // Ensure background color is applied on container
        container.style.background = nextPanel.bg || "#000";

        // If nothing changed structurally, keep the DOM and do minimal in-place updates
        if (samePanel(prevPanel, nextPanel)) {
            // Only case we want to touch is text updates (no animation)
            if (nextPanel.type === "text") {
                const existing = container.querySelector(".text");
                if (existing && existing.textContent !== (nextPanel.text || "")) {
                    existing.textContent = nextPanel.text || "";
                }
            }
            return;
        }

        // Structural change (type, src, fit, bg)
        // Decide whether to animate: yes for image/url *source* changes; not for text
        const animate =
            nextPanel.type === "image" ||
            nextPanel.type === "url";

        // Nuke old children and insert fresh content
        container.innerHTML = "";
        const content = createContentEl(nextPanel);

        // If we don't want animation (text), remove the fade class we added
        if (!animate && content.classList.contains("fade")) {
            content.classList.remove("fade");
        }

        container.appendChild(content);
    }

    // Main render entrypoint
    function render(payload) {
        // Apply layout only if it changed
        if (!prev || !sameLayout(prev.layout, payload.layout)) {
            applyLayout(payload.layout || "split");
        }

        // Update slots independently
        renderSlot("top", payload.top, prev ? prev.top : null);
        renderSlot("bottom", payload.bottom, prev ? prev.bottom : null);

        // Keep a frozen snapshot to compare next time
        prev = deepClone(payload);
    }

    // ---- WebSocket wiring ----
    function connect() {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        const url = `${proto}://${location.host}/ws`;
        ws = new WebSocket(url);

        ws.onopen = () => setBadge("Connected", "ok");

        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === "render" && msg.payload) {
                    render(msg.payload);
                }
            } catch (e) {
                console.error("Bad WS message", e);
            }
        };

        ws.onclose = () => {
            setBadge("Disconnected", "error");
            // Retry with backoff
            setTimeout(connect, 1000);
        };

        ws.onerror = () => {
            try { ws.close(); } catch (_) { }
        };
    }

    // First load: fetch initial state (fast paint) then connect WS
    async function boot() {
        try {
            const res = await fetch("/api/state", { cache: "no-store" });
            const json = await res.json();
            if (json) render(json);
        } catch (e) {
            console.warn("Initial state fetch failed", e);
        } finally {
            connect();
        }
    }

    document.addEventListener("DOMContentLoaded", boot);
})();
