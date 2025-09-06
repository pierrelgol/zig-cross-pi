from flask import Flask, send_file, render_template_string
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

ppm_image = "/tmp/frame.ppm"

# Minimal HTML just for the PPM viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPM Image Viewer</title>
    <script>
    function refreshImage() {
        const img = document.getElementById("ppm");
        img.src = "/ppm?t=" + new Date().getTime(); // cache-busting
    }
    setInterval(refreshImage, 200); // refresh every 200ms
    </script>
</head>
<body>
    <h1>PPM Converted Image</h1>
    <img id="ppm" src="/ppm" alt="PPM Converted" style="max-width: 100%;">
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ppm')
def serve_ppm():
    """Convert /tmp/frame.ppm to JPEG on the fly and serve it."""
    if not os.path.exists(ppm_image):
        return "PPM image not found", 404

    try:
        with Image.open(ppm_image) as im:
            buf = BytesIO()
            im.save(buf, format="JPEG")
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return f"Error converting PPM: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
