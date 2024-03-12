import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#wczytywanie datasetu
df = pandas.read_csv('hou_all.csv')

#utworzenie odpowiednich naglowkow kolumn
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT', 'MEDV','BIAS_COL']

#wyswietlenie przykladowo 5 pierwszych wierszy
df.head()

#przygotowywanie danych do ich trenowania
x =  df.drop("MEDV", axis = 1)
y = df["MEDV"]

print(x.shape)
print(y.shape)

#podzial danych na zestawy treningowe i testowe
#dane dzielone sa w proporcji 80 : 20
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 5)

#sprawdzenie podzialu danych
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#utworzenie modelu regresji liniowej i trenowanie go na zestawie treningowym
m1 = LinearRegression()
m1.fit(X_train, y_train)

#utworzenie metody do ewaluacji skutecznosci modeli
def modelEvaluation(y_test, y_pred):
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2s = r2_score(y_test,y_pred)
    print("MSE : ",mse)
    print("RMSE : ",rmse)
    print("R2 : ",r2s)

#ewaluacja modelu regresji liniowej na zestawie tstowym
y_pred = m1.predict(X_test)

modelEvaluation(y_test, y_pred)

#utworzenie modelu regresji wielomianowej
poly = PolynomialFeatures(degree=2)

#transformacja danych treningowych
X_train_poly = poly.fit_transform(X_train)

#trenowanie modelu na przetransformowanych danych
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)

#ewaulacja modelu na danych treningowych
y_test_predicted = poly_reg_model.predict(poly.fit_transform(X_test))
modelEvaluation(y_test, y_test_predicted)

#proba optymalizacji modelu liniowego
m2 = LinearRegression(fit_intercept = False, positive = True, copy_X = False).fit(X_train, y_train)

y_pred2 = m2.predict(X_test)

modelEvaluation(y_test, y_pred2)

#proba optymalizacji modelu wielomianowego
poly2 = PolynomialFeatures(degree = 2)

X_train_poly2 = poly2.fit_transform(X_train)

poly_reg_model2 = LinearRegression(fit_intercept = False, positive = False, copy_X = False).fit(X_train_poly2, y_train)

y_test_predicted2 = poly_reg_model2.predict(poly.fit_transform(X_test))
modelEvaluation(y_test, y_test_predicted2)
