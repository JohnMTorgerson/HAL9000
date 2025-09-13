// Parses ?center=lat,lon&zoom=14&markers=<urlencoded-JSON>
function parseParams() {
  const p = new URLSearchParams(window.location.search);
  const centerStr = p.get("center") || "";
  const zoom = parseInt(p.get("zoom") || "14", 10);
  let center = null;
  if (centerStr.includes(",")) {
    const [lat, lon] = centerStr.split(",").map(Number);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      center = [lat, lon];
    }
  }
  let markers = [];
  const m = p.get("markers");
  if (m) {
    try {
      markers = JSON.parse(decodeURIComponent(m));
    } catch {}
  }
  return { center, zoom, markers };
}

function init() {
  const { center, zoom, markers } = parseParams();

  // Fallback if no center provided: use first marker
  const defaultCenter = center || (markers[0] ? [markers[0].lat, markers[0].lon] : [0, 0]);
  const defaultZoom = Number.isFinite(zoom) ? zoom : 14;

  const map = L.map("map", {
    center: defaultCenter,
    zoom: defaultZoom,
    zoomControl: true,
    attributionControl: true
  });

  // OSM tile layer — for heavier use, consider a key-based provider (MapTiler, Mapbox, etc.)
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20,
    attribution: "© OpenStreetMap contributors"
  }).addTo(map);

  // Add markers
  const latlngs = [];
  markers.forEach(m => {
    if (!Number.isFinite(m.lat) || !Number.isFinite(m.lon)) return;
    const ll = [m.lat, m.lon];
    latlngs.push(ll);
    const label = m.label || "";
    const popup = (m.popupHtml)
      ? m.popupHtml
      : `<strong>${label}</strong>${m.distance ? `<br>${m.distance}` : ""}`;
    L.marker(ll).addTo(map).bindPopup(popup);
  });

  // Fit bounds if multiple markers; otherwise use provided center/zoom
  if (latlngs.length > 1) {
    const bounds = L.latLngBounds(latlngs);
    map.fitBounds(bounds.pad(0.15));
  }
}

document.addEventListener("DOMContentLoaded", init);
