from PIL import Image, ImageDraw, ImageFont

def create_placeholder(path, text):
    img = Image.new('RGB', (800, 600), color = (240, 240, 240))
    d = ImageDraw.Draw(img)
    # Draw a rectangle/architectural look
    d.rectangle([50, 50, 750, 550], outline="black", width=3)
    d.text((400, 300), text, fill=(0, 0, 0), anchor="mm")
    d.text((400, 350), "(Original asset not available)", fill=(100, 100, 100), anchor="mm")
    img.save(path)

if __name__ == '__main__':
    create_placeholder('Report/images/tardal_arch.png', 'TarDAL Architecture Diagram')
    create_placeholder('Report/images/msrs_result.png', 'Baseline Inference Result')
