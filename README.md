# TestAssigment
В папке `notebooks` находится EDA и итоговые результаты. В папке `src` класс `DataLoader` для предоработки данный и функции для вычисления метрик и построения графиков.
## Цель
С помощью датасета с несбалансированным числом классов обучить модель машинного обучения предсказывать переменную `default`
## Результаты
Были применены 2 модели: логистическая регрессия и градиентный бустинг.

### Логистическая регрессия
```
ROC_AUC:   0.901
Gini:      0.801
F1_score:  0.227
Log_loss:  0.918

Classification_report: 
               precision    recall  f1-score   support

           0       0.98      0.99      0.99     11759
           1       0.36      0.17      0.23       283
    accuracy                           0.97     12042
   macro avg       0.67      0.58      0.61     12042
weighted avg       0.97      0.97      0.97     12042
```


![image](https://user-images.githubusercontent.com/66497711/170164591-357f8899-8db2-4eab-a071-b5cb51185c88.png)
![image](https://user-images.githubusercontent.com/66497711/170164611-b31815f8-3753-419b-a437-2688fb35f34e.png)


### Градиетный бустинг:
```
ROC_AUC:   0.892
Gini:      0.784
F1_score:  0.063
Log_loss:  0.513
Classification_report: 
               precision    recall  f1-score   support

           0       1.00      0.99      0.99     11982
           1       0.05      0.10      0.06        60

    accuracy                           0.99     12042
   macro avg       0.52      0.54      0.53     12042
weighted avg       0.99      0.99      0.99     12042
```
![image](https://user-images.githubusercontent.com/66497711/170164399-996e8e22-2a07-460e-a334-2071b1a2a1f8.png)
![image](https://user-images.githubusercontent.com/66497711/170164415-707ced79-d7c3-41c4-8328-89734d0d7de1.png)
