from PIL import Image, ImageDraw, ImageFont

def draw_round_rect(draw, box, fill, outline, text):
    draw.rectangle(box, fill=fill, outline=outline, width=2)
    # text centering
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx, cy = box[0] + w/2, box[1] + h/2
    draw.text((cx, cy), text, fill="black", anchor="mm")

def create_arch_diagram(path):
    W, H = 800, 400
    img = Image.new('RGB', (W, H), color='white')
    d = ImageDraw.Draw(img)
    
    # Blocks
    # IR Input
    draw_round_rect(d, [50, 50, 150, 150], "#ffcccc", "red", "Infrared")
    # VI Input
    draw_round_rect(d, [50, 250, 150, 350], "#ccccff", "blue", "Visible")
    
    # Encoder / Concat
    d.line([(150, 100), (200, 200)], fill="black", width=2)
    d.line([(150, 300), (200, 200)], fill="black", width=2)
    draw_round_rect(d, [200, 150, 300, 250], "#e0e0e0", "black", "Encoder\n&\nDense")
    
    # Coordinate Attention (The New Part)
    d.line([(300, 200), (350, 200)], fill="black", width=2)
    
    # CA Box
    d.rectangle([350, 120, 550, 280], outline="green", width=3)
    d.text((450, 135), "Coordinate Attention", fill="green", anchor="mm")
    
    # X/Y Pools
    draw_round_rect(d, [370, 160, 440, 200], "#ccffcc", "green", "X-Pool")
    draw_round_rect(d, [370, 220, 440, 260], "#ccffcc", "green", "Y-Pool")
    
    d.text((500, 210), "Weights", fill="black", anchor="mm")
    
    # Fusion
    d.line([(550, 200), (600, 200)], fill="black", width=2)
    draw_round_rect(d, [600, 150, 700, 250], "#ffe0cc", "orange", "Fusion\nLayer")
    
    # Output
    d.line([(700, 200), (750, 200)], fill="black", width=2)
    d.text((750, 200), "Fused Img", fill="black", anchor="lm")
    
    img.save(path)

if __name__ == '__main__':
    create_arch_diagram('Report/images/arch_improved.png')
