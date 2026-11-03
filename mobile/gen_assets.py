# -*- coding: utf-8 -*-
"""
生成 App 图标和启动页素材。

用法：
    cd mobile
    python gen_assets.py

会生成：
- www/icons/icon-192.png
- www/icons/icon-512.png
- www/icons/icon-152.png
- android/resources/drawable-*/*.png（Capacitor 需要）
- ios/App/AppAssets/... （Capacitor 需要）

依赖：pip install Pillow
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent

# 品牌色（跟 index.html 里的主题色一致）
PRIMARY = (22, 119, 255)         # 支付宝蓝
PRIMARY_DARK = (11, 77, 180)
BG_LIGHT = (245, 247, 252)


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """尝试加载系统中文字体，找不到就退回默认字体。"""
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑 Bold
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_icon(size: int) -> Image.Image:
    """生成一个圆角方形图标：蓝底 + 白色"账"字。"""
    img = Image.new("RGBA", (size, size), PRIMARY)
    draw = ImageDraw.Draw(img)

    # 顶部深色渐变带（视觉层次）
    band_height = int(size * 0.32)
    for y in range(band_height):
        alpha = int(60 * (1 - y / band_height))
        draw.line([(0, y), (size, y)], fill=(0, 0, 0, alpha))

    # 中心大字"账"
    font_size = int(size * 0.62)
    font = _try_font(font_size)
    text = "账"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (size - tw) / 2 - bbox[0]
        ty = (size - th) / 2 - bbox[1] - int(size * 0.02)
    except Exception:
        tw, th = size * 0.5, size * 0.5
        tx, ty = size * 0.25, size * 0.25

    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    # 底部小圆点装饰（表示记账点）
    dot_r = int(size * 0.03)
    dot_y = int(size * 0.88)
    for dx in (-int(size * 0.15), 0, int(size * 0.15)):
        cx = size // 2 + dx
        draw.ellipse(
            [cx - dot_r, dot_y - dot_r, cx + dot_r, dot_y + dot_r],
            fill=(255, 255, 255),
        )

    return img


def make_splash(width: int, height: int) -> Image.Image:
    """生成启动页：浅色背景 + 中心 Logo + 底部品牌色条。"""
    img = Image.new("RGB", (width, height), BG_LIGHT)
    draw = ImageDraw.Draw(img)

    # 中心 Logo（缩小版图标）
    icon_size = min(width, height) // 3
    icon = make_icon(icon_size).convert("RGB")
    x = (width - icon_size) // 2
    y = (height - icon_size) // 2 - int(height * 0.08)
    img.paste(icon, (x, y))

    # App 名称文字
    font_big = _try_font(int(icon_size * 0.15))
    title = "AI 家庭记账"
    try:
        bbox = draw.textbbox((0, 0), title, font=font_big)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (width - tw) / 2 - bbox[0]
        ty = y + icon_size + int(height * 0.03)
    except Exception:
        tw, th = width * 0.3, height * 0.03
        tx, ty = width * 0.35, y + icon_size + 20

    draw.text((tx, ty), title, fill=PRIMARY_DARK, font=font_big)

    # 底部品牌色条
    bar_h = int(height * 0.02)
    draw.rectangle([0, height - bar_h, width, height], fill=PRIMARY)

    return img


def install_to_platform_dirs():
    """把图标和启动页放到 Capacitor 期望的目录里。

    Android 侧按屏幕密度分目录（xhdpi/xxhdpi/xxxhdpi）。
    iOS 侧统一放到 AppIcon.appiconset 和 LaunchScreen 的 Contents.json。
    """
    # ===== Android =====
    android_base = OUT_DIR / "android" / "app" / "src" / "main" / "res"
    densities = {
        "drawable-mdpi": 48,
        "drawable-hdpi": 72,
        "drawable-xhdpi": 96,
        "drawable-xxhdpi": 144,
        "drawable-xxxhdpi": 192,
    }
    for folder, size in densities.items():
        d = android_base / folder
        d.mkdir(parents=True, exist_ok=True)
        make_icon(size).save(d / "icon.png")

    # 启动页图片（各密度）
    splash_sizes = {
        "drawable-mdpi": (1080, 1920),
        "drawable-hdpi": (1620, 2880),
        "drawable-xhdpi": (2160, 3840),
        "drawable-xxhdpi": (3240, 5760),
    }
    for folder, (w, h) in splash_sizes.items():
        d = android_base / folder
        d.mkdir(parents=True, exist_ok=True)
        make_splash(w, h).save(d / "splash.png")

    # ===== iOS =====
    ios_assets = OUT_DIR / "ios" / "App" / "App" / "Assets"
    ios_icon_dir = ios_assets / "AppIcon.appiconset"
    ios_launch_dir = ios_assets / "LaunchScreen"
    ios_icon_dir.mkdir(parents=True, exist_ok=True)
    ios_launch_dir.mkdir(parents=True, exist_ok=True)

    ios_icon_sizes = [20, 29, 40, 58, 60, 76, 80, 87, 120, 152, 167, 180, 1024]
    for s in ios_icon_sizes:
        make_icon(s).save(ios_icon_dir / f"icon_{s}x{s}.png")

    # iOS 启动图（iPhone 主流尺寸）
    for name, (w, h) in {
        "screen-667h": (750, 1334),
        "screen-736h": (828, 1792),
        "screen-667h_Landscape": (1334, 750),
    }.items():
        make_splash(w, h).save(ios_launch_dir / f"image-{name}.png")


def main():
    # PWA / WebView 用
    pwa_icons = OUT_DIR / "www" / "icons"
    pwa_icons.mkdir(parents=True, exist_ok=True)
    for size, filename in [(192, "icon-192.png"), (512, "icon-512.png"), (152, "icon-152.png")]:
        make_icon(size).save(pwa_icons / filename)

    # 平台资源
    install_to_platform_dirs()

    print("[OK] 图标和启动页生成完成")
    print(f"  Web 图标:    {pwa_icons}")
    print(f"  Android 图标: {OUT_DIR / 'android' / 'app' / 'src' / 'main' / 'res'}")
    print(f"  iOS 图标:     {OUT_DIR / 'ios' / 'App' / 'App' / 'Assets'}")
    print()
    print("提示：如果之后想换图标，改上面的 PRIMARY 颜色和文字参数，重新跑这个脚本。")


if __name__ == "__main__":
    main()
