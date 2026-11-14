"""Draws the sombrero app icon. Run once: ./venv/bin/python make_icon.py"""
from PIL import Image, ImageChops, ImageDraw

S = 2048  # draw big, then shrink for smooth edges
k = S / 1024

def box(cx, cy, rx, ry):
    return [int((cx - rx) * k), int((cy - ry) * k), int((cx + rx) * k), int((cy + ry) * k)]

def mask():
    return Image.new("L", (S, S), 0)

def ellipse_mask(cx, cy, rx, ry):
    m = mask(); ImageDraw.Draw(m).ellipse(box(cx, cy, rx, ry), fill=255); return m

def rect_mask(x0, y0, x1, y1):
    m = mask(); ImageDraw.Draw(m).rectangle([int(x0 * k), int(y0 * k), int(x1 * k), int(y1 * k)], fill=255); return m

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
def paint(m, color):
    img.paste(Image.new("RGBA", (S, S), color), (0, 0), m)

# Background tile
d.rounded_rectangle(box(512, 512, 412, 412), radius=int(185 * k), fill="#12807c")
d.rounded_rectangle(box(512, 500, 412, 400), radius=int(185 * k), fill="#1aa39d")

STRAW, STRAW_DARK, STRAW_LIGHT = "#e9b949", "#b98420", "#f6d77a"
RED, GREEN, WHITE = "#d7263d", "#1f8a4c", "#fff7e0"

# Brim: dark underside, then the top surface a little higher
paint(ellipse_mask(512, 668, 372, 150), STRAW_DARK)
paint(ellipse_mask(512, 648, 372, 150), STRAW)
ring = ImageChops.subtract(ellipse_mask(512, 648, 340, 130), ellipse_mask(512, 648, 312, 114))
paint(ring, RED)
ring = ImageChops.subtract(ellipse_mask(512, 648, 300, 108), ellipse_mask(512, 648, 286, 100))
paint(ring, GREEN)
paint(ellipse_mask(512, 640, 200, 62), STRAW_DARK)  # shadow under the crown

# Crown: a tall dome with a curved base
CX, BASE_Y, RX = 512, 610, 168
dome = ImageChops.multiply(ellipse_mask(CX, 520, RX, 300), rect_mask(0, 0, 1024, BASE_Y))
crown = ImageChops.lighter(dome, ellipse_mask(CX, BASE_Y, RX, 52))
paint(crown, STRAW)
paint(ImageChops.multiply(crown, ellipse_mask(CX - 60, 400, 70, 150)), STRAW_LIGHT)  # highlight

def below_curve(lift):  # the part of the crown under a curve parallel to the base
    above = ImageChops.lighter(rect_mask(0, 0, 1024, BASE_Y - lift), ellipse_mask(CX, BASE_Y - lift, RX, 52))
    return ImageChops.subtract(crown, above)

paint(below_curve(78), RED)
paint(below_curve(20), GREEN)
band = ImageChops.subtract(below_curve(78), below_curve(20))
dots = mask(); dd = ImageDraw.Draw(dots)
for i in range(-3, 4):
    x = CX + i * 44
    y = BASE_Y - 49 + 52 * (1 - (1 - (i * 44 / RX) ** 2) ** 0.5) * -1 + 52
    dd.regular_polygon((int(x * k), int((y - 4) * k), int(13 * k)), 4, rotation=45, fill=255)
paint(ImageChops.multiply(dots, band), WHITE)

img.resize((1024, 1024), Image.LANCZOS).save("icon.png")
