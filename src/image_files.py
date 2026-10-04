"""Bounded downloads and decoding of image-search results (no content policy)."""
import io
import ipaddress
import socket
import time
import warnings
from urllib.parse import urljoin, urlsplit

import requests

MAX_IMAGE_BYTES = 8 * 1024 * 1024


def web_url(value):
    if not isinstance(value, str) or len(value) > 8192:
        return False
    try:
        url = urlsplit(value)
        return (url.scheme in ('http', 'https') and bool(url.hostname)
                and not url.username and not url.password
                and url.port in (None, 80, 443)
                and not any(ord(c) < 32 for c in value))
    except ValueError:
        return False


def public_url(value):
    """Do not let retrieved URLs address HAL's own machine or private network."""
    if not web_url(value):
        raise ValueError('Invalid image URL')
    url = urlsplit(value)
    addresses = socket.getaddrinfo(url.hostname, url.port or (443 if url.scheme == 'https' else 80),
                                  type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Image URL is not on the public internet')


def normalize_image(data):
    # Lazy import: an unavailable optional image dependency must not break Ollama
    # or HAL startup. Install with requirements-images.txt.
    from PIL import Image, ImageOps
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds download limit')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('JPEG', 'PNG', 'WEBP', 'GIF'):
                raise ValueError('Unsupported image format')
            if min(im.size) < 64 or im.width * im.height > 25_000_000:
                raise ValueError('Image dimensions are unsuitable')
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            im.thumbnail((1600, 1600))
            rgba = im.convert('RGBA')
            rgb = Image.new('RGB', rgba.size, 'white')
            rgb.paste(rgba, mask=rgba.getchannel('A'))
            output = io.BytesIO()
            rgb.save(output, format='JPEG', quality=88)
            return output.getvalue()


def download_image(url):
    deadline = time.monotonic() + 15
    with requests.Session() as session:
        for _ in range(4):
            public_url(url)
            with session.get(url, stream=True, allow_redirects=False, timeout=(4, 6),
                             headers={'User-Agent': 'HAL9000 image viewer',
                                      'Accept': 'image/*'}) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get('Location', ''))
                    continue
                response.raise_for_status()
                if int(response.headers.get('Content-Length', '0')) > MAX_IMAGE_BYTES:
                    raise ValueError('Image exceeds download limit')
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES or time.monotonic() > deadline:
                        raise ValueError('Image download exceeded its budget')
                return normalize_image(bytes(data))
    raise ValueError('Too many image redirects')
