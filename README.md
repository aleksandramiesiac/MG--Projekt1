# MG--Projekt1

## Dla sieci MLP można zdefiniować
* Liczbę warstw i neuronów ukrytych w każdej warstwie (pełne połączenia pomiędzy warstwami)
* Funkcję aktywacji
* Liczbę iteracji
* Wartość współczynnika nauki
* Wartość współczynnika bezwładności
* Rodzaj problemu: klasyfikacja lub regresja

## Co trzeba oddać?
* Aplikacja(plik konfiguracyjny przetwarzająca zbiory zgodne z formatem podanym na stronie
* Możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
* Prezentacja błędu sieci na zbiorze uczącym i testowym w procesie nauki
* Możliwość prezentacji wartości wag w kolejnych iteracjach uczenia
* Wyjaśnienie przyczyn sukcesu lub porażki danego modelu przy zadanych przez prowadzącego parametrach sieci i zadania
* Wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji
* Raport opisujący powstały program, przeprowadzone eksperymenty i wnioski

## TO DO:
- argmax w przypadku klasyfikacji
- dodać do pliku konfiguracyjnego nazwę plików treningowego/testowego i dostosować main do tego (Agata)
- rysowanie wykresów
- raport (jupyter?)
- na koniec: czyszczenie kodu xd (usuwanie zbędnych komentarzy itp.)

opcjonalnie:
- dodać do pliku konfiguracyjnego (liczbę neuronów w każdej warstwie i) współczynnik bezwładności
- dostosować NeuralNetwork do powyższego
