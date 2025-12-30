from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd


df = pd.read_csv("eat.csv")
text_primer = df['eat_type'].tolist()
lable = df['usefuless'].tolist()


text_train, text_test, y_train, y_test = train_test_split(text_primer, lable, test_size=0.33, random_state=56)

pipe = make_pipeline(
    CountVectorizer(), 
    MultinomialNB() 
)
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)

def main():
    print("Здравствуйте, этот искусственный интелект обучен различать фастфуд и здоровую еду")
    data = input("Напишите вашу еду и ИИ определит это фастфуд или нет \n")
    if pipe.predict([data])[0] == 0:
        print(data,"- это фастфуд")
    else:
        print(data,"- это здоровая еда")
    print(f"Точность:{accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()






