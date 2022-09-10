# I.G.O.R.
Imperfect but Glorious Oppressive Robot
![misha_sleeps](images/misha_sleeps.gif)

Система удаленного констроля и анализа поведения сотрудника на рабочем месте. Включает детекцию сна, детекцию эмоций, инструменты обратной связи, сбора статистики и визуализации.
Это проект [@ShakalTabaqui](https://github.com/ShakalTabaqui), [@VAapero1](https://github.com/VAapero1), [@OlgaKrylova](https://github.com/OlgaKrylova), [@datascientist73](https://github.com/datascientist73) в [Elbrus coding bootcamp](https://github.com/Elbrus-DataScience).

Мы использовали библиотеку [MediaPipe](https://google.github.io/mediapipe/)" для детекции лица, позы человека и получения координат точек на лице и теле. На основе информациии о точках мы считали расстояния и углы на лице и теле для обучения классификаторов.
Классификаторов всего три:спящего лица, спящей позы и эмоций. Все они работают на [CatBoost](https://catboost.ai/en/docs/).
Классификатор спящей позы сам по себе не очень информативен: MediaPipe при построении сетки позы определяет только середину глаза одной точкой, поэтому непонятно открыты глаза или нет. Мы использовали его в случаях, когда человек находится в сложной позе и MediaPipe не строит сетку по лицу. Кроме того, мы дополнили его детектором статичности позы (что-то вроде MSE в течение определенного времени). Теперь если человек в странной позе и долго не шевелится - I.G.O.R. считает что он спит. </p> 
Слева - точки которые мы получаем после детекции Mediapipe, на двух других картинках - расстояния и углы, которые мы использовали для классификации эмоций:
<p align="center">
<img src="images/face_annotated.jpg" alt="bash" width="350" height="400"/>
<img src="images/faces_scheme_vit.jpeg" alt="bash" width="320" height="400"/>
<img src="images/face_scheme.jpg" alt="bash" width="320" height="400"/>
</p>

Часть идей по отбору точек для классификации [взята отсюда](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8828335)</p>
Немного картинок, которые мы получали в процессе обучения моделей:
![emo_pic](images/emotions.png)
![mesh_pic](images/mesh.png)

TODO:
- точки для позы
- несколько лиц
- стримлит проверить
- без стримлита реал тайм видео блокнот залить
- про музыку (Влад)
- красоту навести
