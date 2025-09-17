// static/map.js
(() => {
    const qs = new URLSearchParams(location.search);

    // ---- Inputs ---------------------------------------------------------------
    const apiKey = qs.get("key") || window.MAPTILER_API_KEY || "";
    const scale = Math.max(0.5, Math.min(4, Number(qs.get("scale") ?? 2))); // global map element scale (labels/icons/lines)
    const uiScale = Number(qs.get("uiscale") ?? scale);                       // HTML controls (zoom buttons, attribution)
    const pinScale = Number(qs.get("pinscale") ?? scale);                     // our GeoJSON circle pins

    const defaultCenter = [
        Number(qs.get("lng") ?? -93.265), // Minneapolis default
        Number(qs.get("lat") ?? 44.9778),
    ];
    const defaultZoom = Number(qs.get("zoom") ?? 12);

    // We keep the same map bounds/zoom. Instead of zooming out, we
    // reduce detail by hiding “small” layers at the current zoom.
    const detailBoost = Math.max(0, Math.log2(Math.max(1, scale))); // e.g. scale=2 -> +1 “zoom level” of hidden detail

    // ---- Robust pin parsing (pins|markers|points; json|encoded|base64) --------
    function tryParse(str) {
        if (!str) return null;
        try { return JSON.parse(str); } catch (_) { }
        try { return JSON.parse(decodeURIComponent(str)); } catch (_) { }
        try { return JSON.parse(atob(str)); } catch (_) { }
        return null;
    }
    function normalizePin(p) {
        const lng = Number(p.lng ?? p.lon ?? p.longitude);
        const lat = Number(p.lat ?? p.latitude);
        if (!Number.isFinite(lng) || !Number.isFinite(lat)) return null;
        return {
            lng, lat,
            color: p.color || "#8b0000",
            size: Number(p.size ?? p.r ?? 6),
            stroke: p.stroke || "#1a1a1a",
            strokeWidth: Number(p.strokeWidth ?? p.sr ?? 2),
            label: String(p.label ?? p.name ?? p.title ?? ""),  // <- include a label if provided
            isUser: Boolean(p.isUser === true),                 // <- NEW: flag for “my location”
        };
    }
    const rawPins = qs.get("pins") ?? qs.get("markers") ?? qs.get("points");
    let pins = [];
    const parsed = tryParse(rawPins);
    if (Array.isArray(parsed)) pins = parsed.map(normalizePin).filter(Boolean);

    const centerOverride = (
        qs.get("lng") !== null && qs.get("lat") !== null
    ) ? [Number(qs.get("lng")), Number(qs.get("lat"))] : null;

    // ---- Style loading with sprite/glyph key fix ------------------------------
    const styleUrl = `https://api.maptiler.com/maps/streets/style.json?key=${encodeURIComponent(apiKey)}`;

    async function loadStyleEnsuringSprites(url) {
        const style = await (await fetch(url, { cache: "no-cache" })).json();
        const ensureKey = (u) => {
            if (!u) return u;
            if (/\bkey=/.test(u)) return u; // already has key
            return u + (u.includes("?") ? "&" : "?") + "key=" + encodeURIComponent(apiKey);
        };
        style.glyphs = ensureKey(style.glyphs);
        style.sprite = ensureKey(style.sprite);
        return style;
    }

    // ---- Utility to scale numeric values or zoom-stop expressions -------------
    function scaleStops(val, factor) {
        if (val == null) return val;

        // Plain number
        if (typeof val === "number") return val * factor;

        // Legacy {stops:[ [z, value], ... ]}
        if (typeof val === "object" && Array.isArray(val.stops)) {
            return {
                ...val,
                stops: val.stops.map(([z, v]) => [z, (typeof v === "number") ? v * factor : v]),
            };
        }

        // Mapbox expressions: ["interpolate"|"step", ["zoom"], z0, v0, z1, v1, ...]
        if (Array.isArray(val) && (val[0] === "interpolate" || val[0] === "step")) {
            const out = val.slice();
            if (out[0] === "interpolate") {
                // interpolate: [ 'interpolate', ['linear'], ['zoom'], z0, v0, z1, v1, ... ]
                for (let i = 4; i < out.length; i += 2) {
                    if (typeof out[i] === "number") out[i] = out[i] * factor;
                }
            } else { // step
                // step: [ 'step', ['zoom'], v0, z1, v1, z2, v2, ... ]
                if (typeof out[2] === "number") out[2] = out[2] * factor; // base value
                for (let i = 4; i < out.length; i += 2) {
                    if (typeof out[i] === "number") out[i] = out[i] * factor;
                }
            }
            return out;
        }

        // Unknown structure -> leave as-is
        return val;
    }

    // Scale a single layer’s relevant props once.
    function scaleLayerProperties(map, layer) {
        const id = layer.id;
        switch (layer.type) {
            case "symbol": {
                // text size (labels) + icon size (sprites) + halo width
                const ts = map.getLayoutProperty(id, "text-size");
                if (ts != null) map.setLayoutProperty(id, "text-size", scaleStops(ts, scale));

                const isz = map.getLayoutProperty(id, "icon-size");
                if (isz != null) map.setLayoutProperty(id, "icon-size", scaleStops(isz, scale));

                const thw = map.getPaintProperty(id, "text-halo-width");
                if (thw != null) map.setPaintProperty(id, "text-halo-width", scaleStops(thw, scale));
                break;
            }
            case "line": {
                const lw = map.getPaintProperty(id, "line-width");
                if (lw != null) map.setPaintProperty(id, "line-width", scaleStops(lw, scale));
                break;
            }
            case "circle": {
                const cr = map.getPaintProperty(id, "circle-radius");
                if (cr != null) map.setPaintProperty(id, "circle-radius", scaleStops(cr, scale));
                break;
            }
            default:
                // fills, rasters, etc. left unchanged
                break;
        }
    }

    // Raise minzoom for “small details” so they don’t render at the current zoom.
    // This preserves the same bounds while effectively lowering clutter.
    function boostDetailFilters(map, boost) {
        if (boost <= 0) return;

        const layers = map.getStyle().layers || [];
        for (const l of layers) {
            const id = (l.id || "").toLowerCase();

            // Default zoom range
            const origMin = Number.isFinite(l.minzoom) ? l.minzoom : 0;
            const origMax = Number.isFinite(l.maxzoom) ? l.maxzoom : 24;

            // Heuristic: push out “small stuff” more than major features.
            let weight = 1.5;

            // Lots of POIs/addresses/housenumbers are cluttery
            if (id.includes("poi")) weight = 2.0;
            if (id.includes("housenum") || id.includes("address") || id.includes("building-number")) weight = Math.max(weight, 2.0);

            // Minor roads, service roads, paths, tracks
            if (
                id.includes("minor") ||
                id.includes("service") ||
                id.includes("path") ||
                id.includes("track")
            ) weight = Math.max(weight, 1.5);

            // Waterways, contours often add noise at distance
            if (id.includes("waterway") || id.includes("contour")) weight = Math.max(weight, 1.0);

            // MapTiler streets often uses 'transportation-*' ids for roads.
            // Secondary/tertiary get a small push; primaries stay visible.
            if (id.includes("transportation") && (id.includes("secondary") || id.includes("tertiary"))) {
                weight = Math.max(weight, 0.6);
            }

            // Never hide core “place” labels (cities/countries) with this pass.
            if (id.includes("place") && id.includes("label")) {
                weight = Math.max(weight, 0); // i.e., do not boost
            }

            if (weight > 0) {
                const newMin = Math.min(origMax - 0.1, origMin + boost * weight);
                map.setLayerZoomRange(l.id, newMin, origMax);
            }
        }
    }

    // ---- Enlarge built-in UI (zoom buttons, attribution) via CSS --------------
    (function injectUiCss(mult) {
        const css = `
          .maplibregl-ctrl button {
            width:${30 * mult}px; height:${30 * mult}px; font-size:${14 * mult}px;
          }
          .maplibregl-ctrl-group { box-shadow:none; }
          .maplibregl-ctrl-attrib, .maplibregl-ctrl-attrib a {
            font-size:${12 * mult}px;
          }
        `;
        const styleEl = document.createElement("style");
        styleEl.textContent = css;
        document.head.appendChild(styleEl);
    })(uiScale);

    // ---- Init -----------------------------------------------------------------
    (async function init() {
        const style = await loadStyleEnsuringSprites(styleUrl);

        const map = new maplibregl.Map({
            container: "map",
            style,
            center: defaultCenter,
            zoom: defaultZoom,
            attributionControl: false,
            hash: false,
            cooperativeGestures: true,
        });

        map.addControl(new maplibregl.AttributionControl({ compact: false }), "bottom-left");
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

        map.on("load", () => {
            // 1) Scale labels/icons/lines ONCE (avoid compounding)
            const layers = (map.getStyle().layers || []);
            layers.forEach(l => scaleLayerProperties(map, l));

            // 2) Reduce clutter at SAME zoom by raising minzoom on small-details layers
            boostDetailFilters(map, detailBoost);

            // 3) Pins (scaled by pinScale)
            if (pins.length > 0) {
                const features = pins.map((p, i) => ({
                    type: "Feature",
                    id: i,
                    geometry: { type: "Point", coordinates: [p.lng, p.lat] },
                    properties: {
                        color: p.color,
                        r: p.size,
                        stroke: p.stroke,
                        sr: p.strokeWidth,
                        label: p.label,        // <- carry label through to GeoJSON feature
                        isUser: p.isUser === true, // <- NEW
                    },
                }));

                map.addSource("pins", {
                    type: "geojson",
                    data: { type: "FeatureCollection", features },
                });

                // Stroke ring (darker for user)
                map.addLayer({
                    id: "pins-stroke",
                    type: "circle",
                    source: "pins",
                    paint: {
                        "circle-color": ["case",
                            ["==", ["get", "isUser"], true], "#0b3d91",                           // user ring (dark blue)
                            ["coalesce", ["get", "stroke"], "#1a1a1a"]                              // default ring
                        ],
                        "circle-radius": ["+",
                            ["*", ["case",
                                ["==", ["get", "isUser"], true], ["*", ["get", "r"], 1.2],         // user slightly bigger
                                ["get", "r"]
                            ], pinScale],
                            ["*", ["get", "sr"], pinScale]
                        ],
                        "circle-opacity": 0.9,
                    },
                });

                // Fill (blue for user)
                map.addLayer({
                    id: "pins-fill",
                    type: "circle",
                    source: "pins",
                    paint: {
                        "circle-color": ["case",
                            ["==", ["get", "isUser"], true], "#2e9afe",                            // user fill (blue)
                            ["coalesce", ["get", "color"], "#8b0000"]                               // default fill
                        ],
                        "circle-radius": ["*", ["case",
                            ["==", ["get", "isUser"], true], ["*", ["get", "r"], 1.2],
                            ["get", "r"]
                        ], pinScale],
                        "circle-opacity": 0.95,
                    },
                });

                // Labels kept close to pins
                const labelOffsetEm = Math.max(0.12, 0.42 / Math.max(0.5, pinScale));

                map.addLayer({
                    id: "pins-labels",
                    type: "symbol",
                    source: "pins",
                    layout: {
                        "text-field": ["coalesce", ["get", "label"], ""],
                        "text-font": ["Noto Sans Regular", "Arial Unicode MS Regular"],
                        "text-size": 12 * Math.max(1, pinScale),
                        "text-anchor": "left",
                        "text-offset": [labelOffsetEm, 0],
                        "text-allow-overlap": true,
                        "text-ignore-placement": true
                    },
                    paint: {
                        "text-color": "#ffffff",
                        "text-opacity": 0.95,
                        "text-halo-color": "#000000",
                        "text-halo-width": 1.5 * Math.max(1, pinScale),
                    }
                });

                // Center/fit to pins WITHOUT changing zoom for “declutter”.
                if (features.length === 1) {
                    map.setCenter(features[0].geometry.coordinates);
                    if (map.getZoom() < 14) map.setZoom(14); // keep single POI readable
                } else {
                    const b = new maplibregl.LngLatBounds();
                    features.forEach(f => b.extend(f.geometry.coordinates));
                    map.fitBounds(b, { padding: 60 * uiScale, maxZoom: 16, duration: 0 });
                }
            } else if (centerOverride) {
                map.setCenter(centerOverride);
                map.setZoom(defaultZoom);
            }
        });

        // Optional: expose a live pin updater
        window.updatePins = (nextPins) => {
            const normalized = (nextPins || []).map(normalizePin).filter(Boolean);
            const src = map.getSource("pins");
            const data = {
                type: "FeatureCollection",
                features: normalized.map((p, i) => ({
                    type: "Feature",
                    id: i,
                    geometry: { type: "Point", coordinates: [p.lng, p.lat] },
                    properties: {
                        color: p.color,
                        r: p.size,
                        stroke: p.stroke,
                        sr: p.strokeWidth,
                        label: p.label,          // keep label on live updates as well
                        isUser: p.isUser === true, // keep user flag
                    },
                })),
            };
            if (src) src.setData(data);
        };
    })();
})();
