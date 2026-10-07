// static/app.js

(() => {
    const panels = {
        top: document.getElementById("top"),
        bottom: document.getElementById("bottom"),
    };

    const badge = document.getElementById("badge"); // optional status badge

    let ws;
    let wsStartedAt = 0;
    let socketRenderCount = 0;
    let stateRequestPending = false;
    let prev = null; // previous render payload (deep-frozen snapshot)
    let renderVersion = null;
    const retiredServers = new Set();

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
        // Scroll only after attachment: detached nodes have no usable height.
        // This matters when a long transcript replaces the slideshow.
        if (nextPanel.type === 'text') {
            const words = content.querySelector('.text-content');
            words.scrollTop = words.scrollHeight;
        }
        updateCitations(container, nextPanel);
    }

    // Main render entrypoint
    function render(payload) {
        // Both transports can lag. Only accept a newer snapshot from this
        // server, and never return to an old server instance after a restart.
        if (payload.server_id && Number.isSafeInteger(payload.revision)) {
            if (renderVersion) {
                if (payload.server_id === renderVersion.server) {
                    if (payload.revision <= renderVersion.revision) return;
                } else {
                    if (retiredServers.has(payload.server_id)) return;
                    retiredServers.add(renderVersion.server);
                }
            }
            renderVersion = {server: payload.server_id, revision: payload.revision};
        }
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

    // ---- Display connection and recovery ----
    function connect() {
        // A socket can stay CONNECTING/CLOSING without delivering onclose for
        // a long time after a server restart. Do not depend on that event to
        // schedule the next attempt, and ignore callbacks from replaced sockets.
        if (ws && (ws.readyState === WebSocket.OPEN ||
                   (ws.readyState === WebSocket.CONNECTING &&
                    performance.now() - wsStartedAt < 3000))) return;
        const previous = ws;
        ws = null;
        if (previous) {
            try { previous.close(); } catch (_) { }
        }
        const proto = location.protocol === "https:" ? "wss" : "ws";
        const url = `${proto}://${location.host}/ws`;
        let socket;
        try {
            socket = new WebSocket(url);
        } catch (_) {
            return; // The periodic connection check will try again.
        }
        ws = socket;
        wsStartedAt = performance.now();

        socket.onopen = () => {
            if (socket === ws) setBadge("Connected", "ok");
        };

        socket.onmessage = (evt) => {
            if (socket !== ws) return;
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === "render" && msg.payload) {
                    render(msg.payload);
                    socketRenderCount += 1;
                }
            } catch (e) {
                console.error("Bad WS message", e);
            }
        };

        socket.onclose = () => {
            if (socket === ws) setBadge("Reconnecting", "error");
        };

        socket.onerror = () => {
            if (socket === ws) {
                try { socket.close(); } catch (_) { }
            }
        };
    }

    // WebSocket pushes remain the fast path. Periodic HTTP snapshots also
    // update both panes when a socket is missing or silently stops delivering.
    async function refreshState() {
        if (stateRequestPending) return;
        stateRequestPending = true;
        const before = socketRenderCount;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);
        try {
            const res = await fetch("/api/state", { cache: "no-store", signal: controller.signal });
            if (!res.ok) throw new Error('Display state unavailable');
            const json = await res.json();
            // Versioned snapshots are ordered by render(), in either direction.
            // Keep the arrival-count guard for older display servers only.
            if (json && ((json.server_id && Number.isSafeInteger(json.revision)) ||
                         before === socketRenderCount)) render(json);
        } catch (_) {
            // HAL may be stopped; keep the last frame and retry next time.
        } finally {
            clearTimeout(timeout);
            stateRequestPending = false;
        }
    }

    function recover() {
        connect();
        refreshState();
    }

    function boot() {
        // Neither transport waits for the other to become ready.
        recover();
        setInterval(connect, 1000);
        setInterval(refreshState, 2000);
        window.addEventListener('online', recover);
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') recover();
        });
    }

    document.addEventListener("DOMContentLoaded", boot);
})();
