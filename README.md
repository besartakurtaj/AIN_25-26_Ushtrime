<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>Universiteti i Prishtinës</p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p>Profesor: Prof. Dr. Kadri Sylejmani</p>
      <p>Asistent: MSc. Labeat Arbneshi</p>
    </td>
 </tr>
</table>


## Përshkrimi i Projektit: Optimizimi i Orarit Televiziv

Ky projekt adreson **Problemin e Planifikimit Televiziv për Hapësira Publike** (TV Channel Scheduling Optimization for Public Spaces) në kuadër të lëndës **Algoritmet e Avancuara**. Objektivi primar është përzgjedhja dhe planifikimi optimal i një nënbashkësie të programeve televizive në kanale të shumta, me qëllim maksimizimin e pikëve totale të shikueshmërisë.

**Kufizimet dhe Qëllimet Kryesore:**

Përveç kufizimeve bazë kohore, problemi përfshin rregulla specifike për të siguruar një përvojë cilësore shikimi:

*   **Time Window Constraint:** Programet duhet të planifikohen strikt brenda intervalit kohor global të përcaktuar (Hapja dhe Mbyllja).
*   **No Overlap Constraint:** Ndalohet rreptësisht mbivendosja kohore e programeve në të njëjtin kanal.
*   **Minimum Duration:** Programet duhet të kenë një kohëzgjatje minimale për t'u konsideruar të vlefshme.
*   **Genre Repetition:** Për të siguruar shumëllojshmëri, ka një kufizim në numrin e programeve të njëpasnjëshme të të njëjtit zhanër.
*   **Priority Blocks:** Blloqe kohore specifike ku vetëm kanale të caktuara kanë prioritet ose lejohen të transmetojnë.
*   **Time Preferences:** Bonuse pikësh për transmetimin e zhanreve të caktuara në orare të preferuara.
*   **Optimization Goal:** Maksimizimi i funksionit objektiv, duke balancuar pikët e programeve me penalitetet e mundshme.

## Beam Search Scheduler

 **Beam Search Scheduler**, është një algoritëm **deterministik** që tejkalon kufizimet e qasjeve standarde **Greedy** përmes eksplorimit paralel të hapësirës së zgjidhjeve.

**Metodologjia:**

1.  **Beam Search Strategy:** Në vend të ndjekjes së një rruge të vetme, algoritmi mirëmban një bashkësi prej $N$ zgjidhjesh të pjesshme më premtuese në çdo hap (**Beam Width**). Kjo mundëson shmangien e minimumeve lokale dhe rikuperimin nga vendimet sub-optimale të hershme.
2.  **Lookahead Mechanism:** Përtej vlerësimit të menjëhershëm, algoritmi implementon një mekanizëm **Lookahead** me thellësi të konfiguruar. Kjo analizon impaktin e vendimeve aktuale në mundësitë e ardhshme, duke parandaluar bllokimin e programeve me vlerë të lartë.
3.  **Density Heuristic:** Për vlerësimin e potencialit të intervaleve kohore të mbetura, përdoret një heuristikë e bazuar në dendësinë e pikëve (pikë/minutë).

**Konfigurimi i Parametrave:**
*   **Beam Width:** 100. Ruan 100 degëzimet më të mira të pemës së kërkimit në çdo iteracion.
*   **Lookahead:** 4 hapa. Vlerëson pasojat e vendimeve deri në 4 nivele thellësi.
*   **Density Percentile:** Fokusohet në 25% të programeve më të mira për vlerësim heuristik)

Ky kombinim i eksplorimit **Beam Search** dhe heuristikave të avancuara **Lookahead** mundëson gjetjen e zgjidhjeve të cilësisë së lartë në mënyrë efikase.

## Ekzekutimi i Projektit

Për të ekzekutuar projektin dhe për të gjeneruar orarin optimal, ndiqni hapat e mëposhtëm:

1.  Sigurohuni që keni të instaluar Python 3.
2.  Hapni terminalin në direktorinë kryesore të projektit.
3.  Ekzekutoni komandën:
    ```bash
    python3 main.py
    ```
4.  Do t'ju shfaqet një listë e fajllave hyrës (input) të disponueshëm (p.sh., `usa_tv_input.json`, `uk_tv_input.json`, etj.).
5.  Shkruani numrin e indeksit që korrespondon me fajllin që dëshironi të procesoni dhe shtypni Enter.

Algoritmi do të fillojë ekzekutimin dhe në fund do të ruajë rezultatin në folderin `data/output/`.

## Genetic Algorithm Scheduler

**Genetic Algorithm Scheduler** është një algoritëm **stokastik** për optimizimin e orarit televiziv përmes mekanizmave të inspiruar nga evolucioni natyror.

**Metodologjia:**

1. **Population Initialization:** Popullata fillestare përmban një zgjidhje të bazuar në Greedy+Lookahead dhe POP_SIZE-1 variante stokastike për të siguruar diversitet.

2. **Selection:** Përdor **Tournament Selection** - zgjidhen dy individë të rastit dhe më i miri shënohet për riproduksion. Kjo siguron presion selektiv ndaj zgjidhjeve më të mira.

3. **Crossover:** Krijon fëmijë të rinj nga dy prindër duke përdorur ndarje kohore të rastit. Pjesa e parë vjen nga prindit 1, pjesa e dytë nga prindit 2, me përshtatje për të siguruar pajtueshmëri të kufizimeve.

4. **Mutation:** Ndryshon një zgjidhje duke e prerë në një pikë të rastit dhe duke ri-mbushur pjesën tjetër stokastikisht. Kjo siguron eksplorimin e hapësirës së zgjidhjeve.

5. **Elitism:** Mban individët më të mirë të vjetër; pjesa tjetër zëvendësohet nga fëmijë të rinj.

6. **Local Improvement:** Pas çdo gjeneratë, zbatohet kërkimi lokal për të rafinuar zgjidhjen më të mirë.

**Konfigurimi i Parametrave:**

Projekti përdor 4 konfigurimi eksperimentale:

| Konfigurimi | POP_SIZE | GENERATIONS | CROSSOVER_RATE | MUTATION_RATE | TOURNAMENT_SIZE | ELITISM |
|------------|----------|-------------|-----------------|-----------------|-----------------|---------|
| experiment_1_small_pop | 5 | 30 | 0.80 | 0.25 | 2 | 1 |
| experiment_2_medium_pop | 10 | 30 | 0.80 | 0.25 | 3 | 1 |
| experiment_3_large_pop | 20 | 20 | 0.85 | 0.20 | 3 | 2 |
| experiment_4_high_mutation | 10 | 30 | 0.70 | 0.40 | 2 | 1 |

**TIME_LIMIT:** 300 sekonda për çdo run

## Rezultatet e Ekzekutimit Batch 

**Koha Totale e Ekzekutimit:** 39.95 sekonda  
**Instancat e Testuar:** 17 grupe të dhënash  
**Konfigurimi Totali:** 4 × 17 = 68 variacionet e parametrave

### Rezultatet më të Mirë për Çdo Instancë

| Instanca | Konfigurimi Optimal | Koha (s) |
|----------|-------------------|----------|
| australia_iptv.json | experiment_1_small_pop | 0.25 |
| canada_pw.json | experiment_4_high_mutation | 0.20 |
| china_pw.json | experiment_4_high_mutation | 0.31 |
| croatia_tv_input.json | experiment_3_large_pop | 0.10 |
| france_iptv.json | experiment_3_large_pop | 0.15 |
| germany_tv_input.json | experiment_4_high_mutation | 0.10 |
| kosovo_tv_input.json | experiment_4_high_mutation | 0.10 |
| netherlands_tv_input.json | experiment_3_large_pop | 0.11 |
| singapore_pw.json | experiment_4_high_mutation | 0.13 |
| spain_iptv.json | experiment_4_high_mutation | 0.14 |
| toy.json | experiment_2_medium_pop | 0.10 |
| uk_iptv.json | experiment_2_medium_pop | 0.21 |
| uk_tv_input.json | experiment_1_small_pop | 0.12 |
| us_iptv.json | experiment_1_small_pop | 2.23 |
| usa_tv_input.json | experiment_4_high_mutation | 0.27 |
| youtube_gold.json | experiment_1_small_pop | 2.36 |
| youtube_premium.json | experiment_3_large_pop | 2.99 |

### Përfundimet Kryesore

- **experiment_4_high_mutation** performon më mirë në 7 instanca (41% e përgjithshme)
- **experiment_3_large_pop** më e mirë në 4 instanca (24%)
- **experiment_1_small_pop** më e mirë në 4 instanca (24%)
- **experiment_2_medium_pop** më e mirë në 2 instanca (12%)

**Gjetje:** Konfigurimi me **mutation të lartë** është më efektiv në shumicën e rasteve, veçanërisht për instancat me kompleksitet mesatar.

## Ekzekutimi i Projektit

### Për Një Instancë të Vetme

```bash
python main_new.py
```

Zgjidhni:
1. Fajllin hyrës (input)
2. Scheduler: `3` për Genetic Algorithm
3. Konfigurimin GA (p.sh., `experiment_2_medium_pop`)

### Për Të Ekzekutuar të Gjithë Instancat me të Gjithë Konfigurimet

```bash
python run_all_experiments.py
```

Ky skript:
- Teston të gjitha 17 instancat
- Ekzekuton secilën me të 4 konfigurimet
- Bën 10 run-e për çdo instancë/konfiguracion
- Ruaj rezultatet individuale në `results/ga_results_[instanca]_[timestamp].json`
- Ruaj rezultatet agregate në `results/batch_results_[timestamp].json`

**Koha Totale:** ~5 minuta për të gjithë variacionet e parametrave


# Krahasimi i Algoritmit Gjenetik, Local Search dhe Hybrid Optimization

Për secilën instancë:

- Janë realizuar 10 ekzekutime të pavarura
- Janë mbledhur statistika të performancës
- Algoritmet janë krahasuar sipas:
  - Rezultatit më të mirë
  - Mesatares së rezultateve
  - Rezultatit më të dobët
  - Devijimit standard
  - Kohës së ekzekutimit
  - Përqindjes së përmirësimit

---

# Algoritmet e Implementuara

## 1. Algoritmi Gjenetik (GA)

Algoritmi Gjenetik është një metaheuristikë e bazuar në evolucionin natyror.

### Komponentët Kryesorë

- Inicializimi i popullatës
- Vlerësimi i fitness-it
- Tournament selection
- Crossover
- Mutation
- Elitism

### Avantazhet

- Eksplorim i mirë i hapësirës së kërkimit
- Ruajtje e diversitetit të popullatës
- Shmangie më e mirë e local optima

### Disavantazhet

- Kosto më e lartë llogaritëse
- Ka nevojë për parametrizim të mirë

---

## 2. Local Search (LS)

Local Search përmirëson një zgjidhje duke eksploruar fqinjët e saj.

### Operatorët e Përdorur

- Swap
- Move
- Replace
- Repair / Refill

### Avantazhet

- Konvergjencë e shpejtë
- Shfrytëzim i fortë i zgjidhjeve
- Implementim i thjeshtë

### Disavantazhet

- Bllokohet lehtë në local optima
- Eksplorim i dobët

---

## 3. Hybrid GA + LS

Qasja Hybrid kombinon:

- Eksplorimin e GA
- Përmirësimin lokal të LS

Qëllimi është që GA të gjejë zona premtuese të hapësirës së kërkimit, ndërsa LS të rafinojë zgjidhjen.

---

# Rezultatet Eksperimentale

# Instanca: `australia_pw`

## Përmbledhja e Rezultateve

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 4135 | 4117.4 | 4114 | 7.03 | 6636.6 s |
| LS | 4114 | 4114 | 4114 | 0.00 | 6760.4 s |
| Hybrid | 4141 | 4123.1 | 4114 | 9.73 | 38907.0 s |

## Analiza

- GA prodhoi zgjidhje të ndryshme dhe cilësore.
- LS konvergoi gjithmonë në rezultatin 4114 dhe uli cilësinë e zgjidhjes.
- Hybrid arriti rezultatin më të mirë (4141).
- Hybrid përmirësoi eksplorimin, por me kosto shumë të lartë ekzekutimi.

## Përfundimi

Hybrid arriti rezultatin më të mirë për këtë instancë, por me kosto shumë të madhe llogaritëse.

---

# Instanca: `canada_pw`

Kanale: 634  
Programe: 12162

## Përmbledhja e Rezultateve

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 4695 | 4695 | 4695 | 0.00 | 1705.9 s |
| LS | 4695 | 4695 | 4695 | 0.00 | 1702.4 s |
| Hybrid | 4695 | 4695 | 4695 | 0.00 | 2091.6 s |

## Analiza

- Të gjitha algoritmet prodhuan të njëjtën zgjidhje.
- Nuk pati përmirësime.
- GA kishte arritur tashmë local optimum.

---

# Instanca: `china_pw`

Kanale: 1254  
Programe: 20429

## Përmbledhja e Rezultateve

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 2657 | 2657 | 2657 | 0.00 | 2466.1 s |
| LS | 2657 | 2657 | 2657 | 0.00 | 1740.4 s |
| Hybrid | 2657 | 2657 | 2657 | 0.00 | 2064.7 s |

## Analiza

- Të gjitha metodat konverguan menjëherë.
- Nuk pati përmirësim të eksplorimit.
- Hapësira e kërkimit duket shumë deterministike.

---

# Instanca: `croatia_tv_input`

Kanale: 15  
Programe: 205

## Përmbledhja e Rezultateve

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 2187 | 2182.1 | 2138 | 14.70 | 2.1 s |
| LS | 2138 | 2138 | 2138 | 0.00 | 156.1 s |
| Hybrid | 2187 | 2187 | 2187 | 0.00 | 49.2 s |

## Analiza

- GA prodhoi zgjidhje të ndryshme.
- LS e përkeqësoi zgjidhjen duke ngecur në local optimum.
- Hybrid ruajti rezultatin më të mirë të GA.

---

# Instanca: `france_iptv`

Kanale: 397  
Programe: 6291

## Përmbledhja e Rezultateve

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 4360 | 4351.1 | 4340 | 9.27 | 2046.2 s |
| LS | 4340 | 4340 | 4340 | 0.00 | 1941.5 s |
| Hybrid | 4359 | 4344.9 | 4340 | 7.76 | 2253.4 s |

## Analiza

- GA ruajti diversitetin dhe arriti rezultatin më të mirë.
- LS përsëri konvergoi para kohe.
- Hybrid stabilizoi rezultatet por nuk e kaloi GA.

---

# Instanca: `germany_tv_input`

Kanale: 5  
Programe: 52

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 1553 | 1553 | 1553 | 0.00 | 2.2 s |
| LS | 1553 | 1553 | 1553 | 0.00 | 235.3 s |
| Hybrid | 1553 | 1553 | 1553 | 0.00 | 82.2 s |

---

# Instanca: `kosovo_tv_input`

Kanale: 13  
Programe: 175

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 2572 | 2572 | 2572 | 0.00 | 3.0 s |
| LS | 2572 | 2572 | 2572 | 0.00 | 330.3 s |
| Hybrid | 2572 | 2572 | 2572 | 0.00 | 110.6 s |

---

# Instanca: `netherlands_tv_input`

Kanale: 12  
Programe: 180

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 2608 | 2608 | 2608 | 0.00 | 3.2 s |
| LS | 2608 | 2608 | 2608 | 0.00 | 384.8 s |
| Hybrid | 2608 | 2608 | 2608 | 0.00 | 128.4 s |

---

# Instanca: `singapore_pw`

Kanale: 211  
Programe: 4158

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 4327 | 4327 | 4327 | 0.00 | 198.6 s |
| LS | 4327 | 4327 | 4327 | 0.00 | 601.6 s |
| Hybrid | 4327 | 4327 | 4327 | 0.00 | 415.8 s |

---

# Instanca: `toy`

Kanale: 3  
Programe: 5

| Algoritmi | Best | Average | Worst | Std Dev | Time |
|---|---|---|---|---|---|
| GA | 380 | 380 | 380 | 0.00 | 0.4 s |
| LS | 380 | 380 | 380 | 0.00 | 1.7 s |
| Hybrid | 380 | 380 | 380 | 0.00 | 1.0 s |






