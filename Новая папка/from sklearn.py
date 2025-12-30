from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
text = [
"Купи айфон за 100 рублей"
"Встреча в офисе в 10.00"
"Вы выиграли миллион пришлите денег"
"Отчёт: продажи за квартал 2025"
"Срочно! Позвони сейчас и получи приз!"
"Напоминание про опалту счёта"

]
lable =[1,0,1,0,1,0]

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size = 0.33, random_state=42 )


pipe = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
pipe.fit(text_train, text_test)
y_pred = pipe.predict(text_test)
print(f"Accuracy {accuracy_score(y_test,y_pred)}")

print("new", pipe.predict())