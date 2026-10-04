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
        if (a.load_token !== b.load_token) return false;

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

    function updateText(el, panel) {
        let words = el.querySelector('.text-content');
        if (!words) {
            words = document.createElement('div');
            words.className = 'text-content';
            el.appendChild(words);
        }
        if (words.textContent !== (panel.text || '')) {
            words.textContent = panel.text || '';
            words.scrollTop = words.scrollHeight;
        }
    }

    function updateCitations(container, panel) {
        let sources = container.querySelector('.citations');
        if (panel.type !== 'image' || !panel.citations?.length) {
            if (sources) sources.remove();
            return;
        }
        if (!sources) {
            sources = document.createElement('div');
            sources.className = 'citations';
            sources.setAttribute('aria-label', 'Image sources');
            container.appendChild(sources);
        }
        const citations = JSON.stringify(panel.citations || []);
        if (sources.dataset.citations === citations) return;
        sources.dataset.citations = citations;
        sources.replaceChildren();
        for (const source of panel.citations || []) {
            try {
                const url = new URL(source.url);
                if (!['https:', 'http:'].includes(url.protocol)) continue;
                const link = document.createElement('a');
                link.href = url.href;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                // Keep the visible credit compact; the destination stays exact.
                link.textContent = url.hostname.replace(/^www\./i, '');
                link.title = url.href;
                sources.appendChild(link);
            } catch (_) { /* Malformed source data is never HTML. */ }
        }
        sources.hidden = !sources.childElementCount;
    }

    function createContentEl(panel) {
        let el;
        if (panel.type === "image") {
            el = document.createElement("img");
            if (panel.load_token) {
                // A one-shot POST could be lost, and animation frames can pause
                // in background tabs. Report decoded images independently of
                // animation, retrying briefly while this exact node is shown.
                const report = (status) => {
                    const send = async (attempt = 0) => {
                        if (!el.isConnected || !el.getClientRects().length) return;
                        const controller = new AbortController();
                        const timeout = setTimeout(() => controller.abort(), 750);
                        try {
                            const response = await fetch('/api/images/loaded', {
                                method: 'POST', headers: {'Content-Type': 'application/json'},
                                signal: controller.signal,
                                body: JSON.stringify({token: panel.load_token, status,
                                                      visibility: document.visibilityState}),
                            });
                            if (!response.ok) throw new Error('Image acknowledgment failed');
                            // A false "ok" means the server no longer wants this
                            // token, so it must not be retried either.
                            await response.json();
                        } catch (_) {
                            if (attempt < 3) setTimeout(() => send(attempt + 1), 250);
                        } finally {
                            clearTimeout(timeout);
                        }
                    };
                    setTimeout(send, 0); // Let renderSlot attach the new element.
                };
                report('received');
                el.onload = () => report(el.naturalWidth ? 'loaded' : 'error');
                el.onerror = () => report('error');
            }
            el.src = panel.src || "";
            el.className = panel.fit === "contain" ? "contain" : "cover";
        } else if (panel.type === "url") {
            el = document.createElement("iframe");
            el.src = panel.src || "about:blank";
        } else {
            el = document.createElement("div");
            el.className = "text";
            updateText(el, panel);
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
            // Update text and credits without replaying the image animation.
            if (nextPanel.type === "text") {
                const existing = container.querySelector(".text");
                if (existing) updateText(existing, nextPanel);
            }
            updateCitations(container, nextPanel);
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
        updateCitations(container, nextPanel);
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
