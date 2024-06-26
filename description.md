## Інструкція та пояснення коду для аналізу задоволеності клієнтів за допомогою CSV-файлів з відгуками

### Огляд

Цей код призначений для аналізу відгуків клієнтів за допомогою методів обробки природної мови (NLP) та машинного навчання. Код дозволяє:
1. Завантажувати CSV-файли з відгуками.
2. Попередньо обробляти дані.
3. Визначати основні терміни у відгуках.
4. Здійснювати тематичне моделювання за допомогою Latent Dirichlet Allocation (LDA).
5. Візуалізувати результати за допомогою діаграм, хмар слів та соціальних графів.

### Кроки алгоритму

1. **Завантаження даних із CSV**:
    - Ми використовуємо бібліотеку `pandas` для завантаження даних із CSV-файлу. Ця бібліотека дозволяє легко маніпулювати та аналізувати дані у вигляді таблиць.
    - Після завантаження файлу ми виводимо список стовпців, щоб переконатися, що файл містить необхідні дані.
    - Ми шукаємо стовпець, який містить текст відгуків. Для цього перевіряємо, чи є слово "text" у назві стовпця (незалежно від регістру).

2. **Попередня обробка даних**:
    - На цьому етапі ми очищаємо текстові дані для подальшого аналізу.
    - Токенізація тексту означає розбиття його на окремі слова (токени).
    - Нормалізація тексту включає перетворення всіх букв у нижній регістр, видалення пунктуації та стоп-слів (часто вживаних слів, таких як "і", "але", "в", які не несуть смислового навантаження).
    - Ми використовуємо бібліотеку `nltk` (Natural Language Toolkit) для виконання цих завдань.

3. **Ідентифікація найбільш часто вживаних термінів**:
    - Після токенізації та очищення тексту ми визначаємо, які слова зустрічаються найчастіше.
    - Використовуючи `Counter` з бібліотеки `collections`, ми рахуємо частоту кожного слова і вибираємо топ-20 найбільш часто вживаних термінів.

4. **Створення матриці термінів-документів**:
    - Ми перетворюємо текстові дані у формат, придатний для машинного навчання.
    - Використовуючи `CountVectorizer` з бібліотеки `sklearn`, ми створюємо матрицю термінів-документів, де рядки представляють документи (в даному випадку окремі відгуки), а стовпці — окремі терміни.
    - Кожна комірка в матриці містить кількість появ конкретного терміна у конкретному документі.

5. **Тематичне моделювання з використанням LDA**:
    - Latent Dirichlet Allocation (LDA) — це статистичний метод для виявлення основних тем у великому обсязі текстових даних.
    - Ми використовуємо LDA для групування термінів у теми на основі їхньої спільної появи у відгуках.
    - Вихід моделі LDA — це список тем, кожна з яких представлена набором термінів із відповідними вагами (ймовірностями).

6. **Візуалізація результатів**:
    - Ми використовуємо різні методи візуалізації для представлення результатів аналізу.
    - Хмара слів показує найбільш часто вживані терміни у відгуках, де розмір слова пропорційний його частоті.
    - Діаграми частотності термінів (бар-чарти) показують, наскільки часто кожен термін зустрічається у текстах.
    - Соціальні графи представляють зв'язки між термінами, де вузли — це терміни, а ребра — їхні співпояви у відгуках.

7. **Фільтрація тем, що стосуються настроїв**:
    - Ми визначаємо теми, які містять ключові слова, пов'язані з настроями (наприклад, "happy", "sad", "angry").
    - Для кожної теми, виявленої LDA, ми перевіряємо, чи містить вона хоча б одне з ключових слів настрою.
    - Теми, які відповідають цим критеріям, вважаються темами настрою і відображаються окремо.

Цей підхід дозволяє проводити глибокий аналіз текстових даних, визначати основні тенденції та настрої серед клієнтів, а також візуалізувати результати для кращого розуміння.

