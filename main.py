from ai_lz import model_main, img

def main():
    models = model_main()   
    for m in models:       
        img(m, "img_2.png")

if __name__ == "__main__":
    main()