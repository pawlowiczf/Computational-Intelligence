# Clonal selection algorithm

Papers:
- The Clonal Selection Algorithm with Engineering Applications ; Leandro Nunes de Castro, Fernando J. Von Zuben ; 2000
https://link.springer.com/article/10.1007/s11390-005-0728-3
https://www.sciencedirect.com/science/article/pii/S0304397510006559
https://cleveralgorithms.com/nature-inspired/immune/clonal_selection_algorithm.html
https://www.cs.unm.edu/~forrest/classes/immuno-class/readings/DeCastro.pdf <- kluczowa publikacja

### Biology

Antygen to zagrożenie (np. wirus).
Antygen to problem optymalizacyjny.

Limfocyt B produkuje przeciwciała.
Każdy limfocyt = jedno potencjalne rozwiązanie problemu.

Affinity maturation (dojrzewanie powinowactwa) to proces immunologiczny, w którym układ odpornościowy „ulepsza” przeciwciała tak, aby coraz silniej i dokładniej wiązały antygen. Na początku odpowiedzi immunologicznej organizm produkuje przeciwciała, które rozpoznają patogen, ale nie są jeszcze „idealnie dopasowane”. W trakcie kilku dni–tygodni następuje ich udoskonalanie, aż powstają warianty o bardzo wysokim powinowactwie (binding affinity).

Jak działa (mechanizm krok po kroku):

1. Somatyczna hipermutacja (SHM)
Limfocyty B mutują geny kodujące przeciwciała (regiony zmienne).
Mutacje są losowe, głównie w miejscach wiążących antygen.
Powstaje wiele wariantów przeciwciał.

2. Selekcja klonalna
Komórki B konkurują o wiązanie antygenu.
Te, które:
wiążą mocniej, przeżywają i proliferują
wiążą słabo, ulegają apoptozie

3. Pomoc limfocytów T
Limfocyty T pomocnicze (Tfh) „zatwierdzają” najlepsze komórki B.
Bez ich sygnałów komórki B nie przetrwają selekcji.

4. Różnicowanie
Wybrane komórki B przekształcają się w:
komórki plazmatyczne → produkują przeciwciała
komórki pamięci → szybka odpowiedź w przyszłości

Antygen – obca cząsteczka, którą układ odpornościowy musi rozpoznać i usunąć.
Limfocyt B – komórka, która rozpoznaje antygen i inicjuje produkcję przeciwciał.
Receptor BCR (na limfocycie B) – „czujnik”, który specyficznie wiąże antygen i uruchamia pierwszy sygnał aktywacji.
Przeciwciało (Ab) – białko produkowane przez limfocyty B, które wiąże antygen i pomaga go neutralizować.
T-helper cell – dostarcza drugi sygnał aktywacji, bez którego limfocyt B nie rozpocznie pełnej odpowiedzi.
Proliferacja – szybkie namnażanie limfocytów B po aktywacji (tworzenie klonów).
Różnicowanie (maturation) – proces, w którym komórki B specjalizują się w różne funkcje.
Komórki plazmatyczne – końcowe komórki produkujące duże ilości przeciwciał.
Komórki pamięci – długowieczne komórki umożliwiające szybką reakcję przy ponownym kontakcie z tym samym antygenem.

Antygen wiąże się z receptorem BCR na powierzchni limfocytu B, co daje pierwszy sygnał aktywacji.
Limfocyt B wchłania antygen, tnie go na fragmenty i prezentuje na swojej powierzchni przy pomocy MHC II.
Limfocyt T pomocniczy rozpoznaje prezentowany fragment i wydziela sygnały chemiczne, dając drugi sygnał aktywacji limfocytowi B.
Aktywowany limfocyt B dzieli się wielokrotnie, tworząc klon identycznych komórek.
Część komórek klonu różnicuje się w komórki plazmatyczne, które produkują i wydzielają przeciwciała do krwi, limfy i tkanek.
Część komórek różnicuje się w komórki pamięci, które pozostają w organizmie i umożliwiają szybką reakcję przy ponownym kontakcie z antygenem.
W procesie dojrzewania przeciwciał (affinity maturation) powstają limfocyty B produkujące przeciwciała o wyższym powinowactwie do antygenu.