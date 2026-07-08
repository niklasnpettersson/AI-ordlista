# AI-ordlista (Svenska)

Denna ordlista riktar sig till dig som redan har god kunskap om AI och maskininlärning men vill fördjupa dig i terminologi, tekniker och forskningsbegrepp. Varje post anger det engelska begreppet med en kort, tekniskt korrekt förklaring på svenska. Använd innehållsförteckningen för att hoppa till ett ämnesområde, eller sök i filen efter ett specifikt begrepp.

## Innehållsförteckning

- [ML-grundläggande](#ml-grundlaggande)
- [Deep Learning](#deep-learning)
- [Transformers & Attention](#transformers-attention)
- [Stora språkmodeller](#stora-sprakmodeller)
- [Fine-tuning & Adaptation](#fine-tuning-adaptation)
- [Alignment & RLHF](#alignment-rlhf)
- [RAG & Retrieval](#rag-retrieval)
- [AI-agenter](#ai-agenter)
- [Multimodal AI](#multimodal-ai)
- [Diffusionsmodeller](#diffusionsmodeller)
- [Computer Vision](#computer-vision)
- [Reinforcement Learning](#reinforcement-learning)
- [Utvärdering & Benchmarks](#utvardering-benchmarks)
- [MLOps](#mlops)
- [Hårdvara & Infrastruktur](#hardvara-infrastruktur)
- [Säkerhet & Etik](#sakerhet-etik)
- [API:er & Ekosystem](#api-er-ekosystem)
- [Matematik & Statistik](#matematik-statistik)
- [Forskningskoncept](#forskningskoncept)
- [NLP & Text](#nlp-text)
- [Optimering & Träning](#optimering-traning)
- [Generativ AI & GANs](#generativ-ai-gans)

## ML-grundläggande

**Supervised Learning** — Lärande från märkta exempel där modellen optimerar en förlustfunktion mot kända målvariabler. Typiska uppgifter är klassificering och regression; träningsetiketterna ger en explicit fel signal som styr generalisering till osedda indata.
**Unsupervised Learning** — Lärande utan etiketter; modellen upptäcker struktur, kluster eller latenta representationer i data. Används för explorativ analys, dimensionsreduktion och generativ modellering när märkning är dyr eller omöjlig.
**Semi-supervised Learning** — Kombinerar liten mängd märkt data med stor omärkt datamängd för att förbättra generalisering. Omärkt data hjälper modellen lära datamannifolden medan de märkta exemplen styr uppgiftsspecifik inlärning.
**Self-supervised Learning** — Modellen skapar egna träningsmål från rådata, t.ex. maskerad token-prediktion eller kontrastiv inlärning. Har blivit standard för förträning av stora modeller eftersom den skalar utan manuell annotation.
**Transfer Learning** — Återanvänder representationer från en förtränad modell på en ny uppgift med mindre data eller beräkning. Tidiga lager fångar generella features medan senare lager finjusteras för måluppgiften.
**Domain Adaptation** — Anpassar en modell tränad i en domän (källdomän) till en annan (måldomän) med skiftad datadistribution. Kräver ofta domän-invarianta representationer eller pseudo-etikettering i måldomänen.
**Feature Engineering** — Manuell eller regelbaserad konstruktion av indatarepresentationer som fångar relevant signal för modellen. Var centralt före deep learning men kvarstår viktigt i tabulära och domänspecifika pipelines.
**Feature Selection** — Urval av delmängd av features som minskar dimensionalitet och overfitting utan att förlora prediktiv kraft. Metoder som Lasso, mutual information och wrapper-baserat urval balanserar modellkomplexitet mot tolkbarhet.
**Bias-Variance Tradeoff** — Balans mellan modellens systematiska fel (bias) och känslighet för träningsvariation (varians). Enklare modeller har högre bias men lägre varians; komplexa modeller inverterar tradeoffen och riskerar overfitting.
**Overfitting** — Modellen memorerar träningsdata och presterar dåligt på osedd data; ofta tecken på för hög kapacitet. Motverkas med regularisering, mer data, enklare modeller eller early stopping baserat på valideringsprestanda.
**Underfitting** — Modellen är för enkel för att fånga datamönster; högt fel både på träning och validering. Lösningar inkluderar rikare features, djupare modeller eller längre träning utan att nå overfitting.
**Regularization** — Tekniker som begränsar modellkomplexitet (L1/L2, dropout, early stopping) för bättre generalisering. Inför implicit prior som straffar extrema parametrar och förbättrar robusthet mot brus i träningsdata.
**Cross-validation** — Uppdelning av data i flera fold för robust uppskattning av generaliseringsprestanda. Minskar variansen i prestandauppskattning jämfört med en enda hold-out-delning.
**k-fold Cross-validation** — Data delas i k delar; varje fold används en gång som validering medan övriga tränar. Ger k oberoende prestandauppskattningar som medelvärderas för stabilare metrik.
**Hold-out Validation** — En fast tränings-/valideringsdelning utan rotation; snabb men mer variabel uppskattning. Vanligt vid stora dataset eller när beräkningskostnaden för k-fold är för hög.
**Train/Validation/Test Split** — Tre väggar: träning för parametrar, validering för hyperparametrar, test för slutgiltig utvärdering. Testsetet får endast användas en gång för att undvika implicit optimering mot det.
**Hyperparameter** — Konfigurationsvärde satt före träning (learning rate, batch size) som inte lärs via gradient. Dessa styr optimeringsdynamik och modellkapacitet utanför själva gradientuppdateringen.
**Hyperparameter Tuning** — Systematisk sökning (grid, random, Bayesian) efter optimala hyperparametrar. Valideringsmetrik styr valet; testsetet reserveras för slutlig opartisk utvärdering.
**Grid Search** — Exhaustiv utvärdering av fördefinierade hyperparameterkombinationer. Enkelt men skalerar dåligt i många dimensioner eftersom antalet kombinationer växer exponentiellt.
**Random Search** — Slumpmässigt urval av hyperparameterkombinationer; ofta effektivare än grid search i höga dimensioner. Bergman et al. visade att viktiga hyperparametrar sällan kräver finmaskig grid-sökning.
**Bayesian Optimization** — Sekventiell modellbaserad sökning som balanserar utforskning och exploatering av hyperparameterrum. Använder surrogate-modell (t.ex. Gaussian process) för att välja nästa lovande kombination.
**Learning Curve** — Graf över modellprestanda vs träningsdatamängd; avslöjar under/overfitting och datanhunger. En växande klyfta mellan tränings- och valideringskurva signalerar overfitting.
**Confusion Matrix** — Tabell över sanna/falska positiva/negativa för klassificering; grund för många mätvärden. Ger fullständig bild av vilka feltyper modellen gör, inte bara aggregerad accuracy.
**Precision** — Andel predikterade positiva som faktiskt är positiva: TP/(TP+FP). Viktigt när falska positiva är dyra, t.ex. spamfilter eller medicinsk screening.
**Recall** — Andel faktiska positiva som hittas: TP/(TP+FN); även känt som sensitivity. Prioriteras när falska negativa är kostsamma, t.ex. sjukdomsdetektion eller säkerhetsintrång.
**F1 Score** — Harmoniskt medelvärde av precision och recall; balanserar båda när klasser är obalanserade. Straffar extrema värden på endera metrik hårdare än aritmiskt medelvärde.
**ROC Curve** — Plot av true positive rate mot false positive rate vid varierande klassificeringströsklar. Visar tradeoff mellan sensitivitet och specifikitet oberoende av vald tröskel.
**AUC-ROC** — Area under ROC-kurvan; sammanfattande mått på diskrimineringsförmåga oberoende av tröskel. Värde 0,5 motsvarar slump; 1,0 perfekt separation mellan klasser.
**PR Curve** — Precision-recall-kurva; mer informativ än ROC vid stark klassobalans. Fokuserar på positiv klassens prestanda där negativ klass dominerar datamängden.
**AUC-PR** — Area under precision-recall-kurvan. Baseline beror på klassfrekvens; mer känsligt än AUC-ROC när positiva exempel är sällsynta.
**Log Loss** — Korsentropiförlust för sannolikhetsutsagor; straffar överdrivet självsäkra felprediktioner. Kräver kalibrerade sannolikheter och är differentierbar för gradientbaserad optimering.
**Calibration** — I hur hög grad predikterade sannolikheter matchar empiriska frekvenser. En modell kan ha hög accuracy men dålig kalibrering om den systematiskt är över- eller underconfident.
**Platt Scaling** — Logistisk regression på modellens råa scores för att kalibrera sannolikheter. Enkel post-hoc-metod som fungerar väl när modellen redan rankar korrekt men har felaktiga absoluta sannolikheter.
**Isotonic Regression** — Icke-parametrisk monoton kalibrering som passar flexibla sannolikhetsmappningar. Mer uttrycksfull än Platt scaling men kräver mer kalibreringsdata och riskerar overfitting.
**Class Imbalance** — Ojämn fördelning av klasser som kan snedvrida träning och mätvärden. Accuracy blir missvisande; precision, recall och F1 ger mer meningsfull bild av prestanda.
**Oversampling** — Ökar representationen av minoritetsklasser, t.ex. via duplication eller SMOTE. Balanserar träningsfördelningen men kan öka risken för overfitting på syntetiska eller duplicerade exempel.
**Undersampling** — Minskar majoritetsklasser för balans; risk att förlora information. Användbart vid mycket stora dataset där majoritetsklassen har överflöd av exemplar.
**SMOTE** — Synthetic Minority Over-sampling Technique; interpolerar syntetiska minoritetsexempel. Skapar nya punkter längs linjer mellan närliggande minoritetsexempel i feature-rymden.
**Cost-sensitive Learning** — Viktar förlust efter klasskostnad för att hantera obalans eller asymmetriska fel. Alternativ till sampling som direkt optimerar affärsmässigt relevant felkostnad.
**Ensemble Learning** — Kombinerar flera modeller för robustare prediktioner än enskilda modeller. Minskar varians genom att aggregera diversifierade prediktorer; olika metoder balanserar bias och varians olika.
**Bagging** — Bootstrap aggregating; tränar parallella modeller på bootstrap-sampel och aggregerar. Random Forest är den mest kända bagging-metoden med extra randomisering per split.
**Boosting** — Sekventiellt ensemble som fokuserar på tidigare modellers fel, t.ex. gradient boosting. Varje ny modell korrigerar residualer från ensemblet hittills; risk för overfitting vid för många svaga lärare.
**Stacking** — Meta-modell lär sig kombinera basmodellers prediktioner. Basmodellernas outputs blir features för en överliggande modell som lär optimal viktning eller icke-linjär kombination.
**Random Forest** — Ensemble av beslutsträd med bagging och slumpmässigt feature-urval per split. Robust, kräver lite hyperparametertuning och hanterar icke-linjära relationer väl på tabulär data.
**Gradient Boosting** — Sekventiellt adderar träd som approximerar negativ gradient av förlust. Varje träd tränas på residualer; learning rate styr bidraget per träd för att undvika overfitting.
**XGBoost** — Optimerad gradient boosting-implementation med regularisering och effektiv träning. Inkluderar L1/L2-straff, parallellisering och hantering av saknade värden; dominerade Kaggle-tävlingar länge.
**LightGBM** — Gradient boosting med leaf-wise trädtillväxt och histogram-baserad split-sökning. Snabbare träning och lägre minneskrav än level-wise metoder; kan overfita på små dataset.
**CatBoost** — Gradient boosting med inbyggd hantering av kategoriska features och ordningsbevarande encoding. Kräver minimal förbehandling av kategoriska variabler och hanterar ordningsskiftade kategorier robust.
**Decision Tree** — Rekursiv partitionering av feature-rymden via binära split-regler. Tolkbara och hanterar icke-linjära gränser men tenderar att overfita utan pruning eller ensemble.
**Random Split** — Slumpmässig uppdelning av features vid varje nod i random forest. Dekorrelerar träd i ensemblet och förbättrar generalisering jämfört med att alla träd ser samma features.
**Support Vector Machine** — Hittar maximal marginal-hyperplan; kan utökas med kernel-trick för icke-linjära gränser. Effektiv i högdimensionella rum och med få träningsexempel relativt feature-dimension.
**Kernel Trick** — Implicit mappning till högdimensionellt rum utan explicit feature-transformation. Beräkningskostnaden skalar med antal träningsexempel, vilket begränsar skalbarhet på stora dataset.
**k-Nearest Neighbors** — Icke-parametrisk metod som klassificerar/regresserar baserat på närmaste träningspunkter. Enkel och utan explicit träningsfas men dyr inferens och känslig för feature-skalning och dimensionalitet.
**Naive Bayes** — Probabilistisk klassificerare som antar feature-oberoende givet klass. Trots förenklade antaganden fungerar ofta bra på textklassificering och som snabb baseline.
**Logistic Regression** — Linjär modell med logistisk länk för binär/multinomial klassificering. Ger kalibrerade sannolikheter och tolkbara koefficienter; baseline för många klassificeringsuppgifter.
**Linear Regression** — Modellerar kontinuerligt mål som linjär kombination av features med minsta kvadrat. Antaganden om linjäritet, homoskedasticitet och oberoende fel; grund för regulariserade varianter.
**Ridge Regression** — Linjär regression med L2-regularisering som krymper koefficienter. Hanterar multikollinearitet och stabiliserar lösningen när features är korrelerade eller fler än exemplar.
**Lasso Regression** — Linjär regression med L1-regularisering som ger sparsity och feature selection. Kan nolla koefficienter helt och producera tolkbara modeller med få aktiva features.
**Elastic Net** — Kombinerar L1- och L2-straff för balanserad regularisering. Användbart när grupper av korrelerade features ska väljas tillsammans, till skillnad från ren Lasso.
**Principal Component Analysis** — Linjär dimensionsreduktion som projicerar till ortogonala riktningar med maximal varians. Bevarar global struktur men kan förlora lokalt meningsfull variation i icke-linjära data.
**t-SNE** — Icke-linjär embeddingsmetod för visualisering av högdimensionell data i 2D/3D. Bra för kluster-visualisering men avstånd mellan kluster är inte meningsfullt; inte för dimensionsreduktion vid inferens.
**UMAP** — Manifold learning för dimensionsreduktion; bevarar både lokal och global struktur bättre än t-SNE i många fall. Snabbare än t-SNE och kan användas för både visualisering och förbehandling.
**K-means Clustering** — Partitionerar data i k kluster genom iterativ uppdatering av centroid och tillledning. Kräver förval av k och antar ungefär sfäriska kluster; känslig för initiering och outliers.
**Hierarchical Clustering** — Bygger klusterträd via agglomerativ eller divisiv länkning. Kräver inte förval av antal kluster; dendrogram visar hierarkisk struktur men skalas dåligt på stora dataset.
**DBSCAN** — Densitetsbaserad klustring som hittar godtyckligt formade kluster och markerar brus som outliers. Kräver inte antal kluster i förväg men är känslig för hyperparametrarna eps och min_samples.
**Gaussian Mixture Model** — Probabilistisk klustring som modellerar data som blandning av Gaussiska komponenter. Ger mjuka klustertilldelningar och kan modellera elliptiska kluster till skillnad från k-means.
**Expectation-Maximization** — Iterativ algoritm för att uppskatta latenta variabler i modeller som GMM. E-steg beräknar förväntade latenta tilldelningar; M-steg maximerar parametrar givet dessa.
**Curse of Dimensionality** — Fenomen där data blir gles i höga dimensioner och avståndsmått förlorar discriminativ kraft. Motiverar dimensionsreduktion, feature selection och regularisering i högdimensionella problem.
**Curriculum Learning** — Träningsstrategi som presenterar exempel i ökande svårighetsordning. Mimiker mänskligt lärande och kan stabilisera träning av komplexa modeller genom gradvis ökad svårighet.
**Active Learning** — Modellen väljer vilka exempel som ska annoteras för maximal informationsvinst. Minskar annotationskostnad genom att fokusera mänsklig insats på mest informativa datapunkter.
**Weak Supervision** — Använder brusiga, heuristiska eller partiella etiketter istället för full manuell annotation. Snorkel-liknande ramverk kombinerar flera svaga signaler till träningsetiketter.
**Label Noise** — Felaktiga eller inkonsistenta etiketter i träningsdata som degraderar inlärning. Robust loss-funktioner, rensning och semi-supervised metoder mitigerar effekten av brusiga etiketter.
**Data Augmentation** — Syntetisk utökning av träningsdata via transformationer som bevarar semantik. Vanligt inom bild (rotation, beskärning) och text (back-translation, synonymbyte) för bättre generalisering.
**Synthetic Data** — Artificiellt genererad data för träning, ofta via simulering eller generativa modeller. Adresserar datam brist, integritet och edge cases men riskerar domain gap mot verklig data.
**Concept Drift** — Förändring av datadistribution eller målrelation över tid i produktion. Kräver kontinuerlig övervakning, omträning eller adaptiva modeller för att bibehålla prestanda.
**Covariate Shift** — Förändring av indatafördelning P(X) medan P(Y|X) är stabil. Viktning av träningsdata eller domain adaptation kan korrigera utan nya etiketter i måldomänen.
**Label Shift** — Förändring av målfördelning P(Y) medan P(X|Y) är stabil. Klasspriorer i träning matchar inte deployment; kan korrigeras med känd eller uppskattad priorförskjutning.
**Prior Shift** — Förändring av klasspriorer mellan träning och deployment. Relaterat till label shift; påverkar optimal tröskel och kalibrering vid inferens.
**Distribution Shift** — Generell term för mismatch mellan tränings- och test-/produktionsfördelning. Inkluderar covariate shift, label shift och concept drift som specialfall.
**OOD Detection** — Out-of-distribution detection; identifierar indata som avviker från träningsfördelningen. Kritiskt för säker deployment där modellen annars kan ge överconfidenta felaktiga prediktioner.
**Anomaly Detection** — Hittar avvikande observationer som inte följer normalt mönster. Används inom fraud detection, industriell övervakning och cybersäkerhet utan märkta avvikelser.
**One-class SVM** — Lär en gräns runt normal data för att flagga avvikelser. Tränas endast på normala exempel; effektivt när avvikelser är sällsynta och svåra att märka.
**Isolation Forest** — Anomaly detection genom att isolera observationer med få slumpmässiga splits. Avvikelser isoleras snabbare (färre splits) än normala punkter; skalar bra till högdimensionell data.
**Autoencoder** — Neuralt nätverk som kodar och avkodar data; rekonstruktionsfel indikerar anomalier. Tränas på normal data; högt rekonstruktionsfel signalerar att indata avviker från träningsfördelningen.
**Inductive Bias** — Inbyggda antaganden i modellarkitektur eller algoritm som styr vilka funktioner som lärs lätt. CNN:er antar lokalitet; transformers antar parallell attention över sekvenser.
**No Free Lunch Theorem** — Inget universellt bästa inlärningsalgoritm över alla möjliga datadistributioner. Motiverar att välja modell och induktiv bias utifrån problemets struktur och datans egenskaper.
**Sample Complexity** — Antal exempel som krävs för att lära en hypotesklass med given noggrannhet. Teoretiskt ramverk som relaterar modellkomplexitet till datamängd som behövs för generalisering.
**PAC Learning** — Probably Approximately Correct framework för inlärbarhet med sannolikhetsgarantier. Formaliserar när en hypotesklass är inlärbar med polynomial sample complexity.
**VC Dimension** — Mått på modellklassens kapacitet relaterat till generaliseringsgränser. Högre VC-dimension tillåter rikare funktioner men kräver mer data för att undvika overfitting.
**Rademacher Complexity** — Empiriskt mått på hypotesklassens rikedom och risk för overfitting. Data-beroende alternativ till VC-dimension som ofta ger tightare generaliseringsgränser i praktiken.
**Generalization Gap** — Skillnad mellan tränings- och testprestanda; indikator på overfitting. Stort gap signalerar att modellen memorerar träningsdata snarare än lär generaliserbara mönster.
**Double Descent** — Fenomen där testfel kan minska igen efter interpolationspunkten när modellen växer. Utmanar klassisk bias-variance tradeoff i överparameteriserade regimer med modern deep learning.
**Interpolation Regime** — Träningsregim där modellen kan interpolera all träningsdata (noll träningsfel). Moderna stora nätverk opererar ofta här; implicit regularisering via SGD förklarar delvis varför generalisering ändå sker.
**Scaling Laws** — Empiriska potenslagar som beskriver hur prestanda skalar med modellstorlek, data och compute. Chinchilla-optimala allokeringar har omdefinierat hur man dimensionerar träning av stora modeller.
**Data-centric AI** — Fokus på att förbättra datakvalitet, annotation och pipeline snarare än enbart modellarkitektur. Andrew Ng och andra betonar att datakvalitet ofta ger större vinster än marginal arkitekturförbättringar.
**Leakage** — Oavsiktlig exponering av test-/målinformation under träning som ger optimistiska mätvärden. Vanliga källor: feature leakage, temporal leakage och duplicerade exempel mellan train och test.
**Target Leakage** — Features som indirekt innehåller målvariabeln och inte finns vid prediktionstid. Ger artificiellt hög prestanda i utvärdering men modellen misslyckas i produktion.
**Train-test Contamination** — Överlapp mellan tränings- och testdata som snedvrider utvärdering. Särskilt vanligt vid web-scraping och duplicerade dokument; kräver deduplicering före split.
**Reproducibility** — Förmåga att återskapa resultat med samma kod, data och seed. Kräver versionshantering av data, kod, miljö och dokumentation av hyperparametrar och slumpfrön.
**Ablation Study** — Systematisk borttagning av komponenter för att mäta deras bidrag till prestanda. Standard i forskning för att isolera vilka designval som faktiskt driver resultatförbättringar.

## Deep Learning

**Neural Network** — Beräkningsgraf av sammankopplade neuroner som approximerar icke-linjära funktioner via viktade transformationer. Tränas med backpropagation och gradient descent; universal approximation theorem garanterar teoretisk uttrycksfullhet.
**Deep Neural Network** — Nätverk med många dolda lager som lär hierarkiska representationer. Djupare lager fångar successivt mer abstrakta mönster; kräver tekniker som residual connections och normalisering för stabil träning.
**Perceptron** — Enkel binär klassificerare: viktad summa följt av tröskelaktivering. Historiskt grundsten; kan endast lära linjärt separabla mönster utan dolda lager eller icke-linearitet.
**Multilayer Perceptron** — Fully connected feedforward-nätverk med ett eller flera dolda lager. Universal approximator med tillräcklig bredd; används som baseline och i tabulära deep learning-uppgifter.
**Activation Function** — Icke-linjär transformation efter linjärt lager; möjliggör universell approximation. Utan den skulle sammansatta lager kollapsa till en enda linjär transformation.
**ReLU** — Rectified Linear Unit: max(0,x); standardaktivering tack vare snabbhet och gradientflöde. Undviker vanishing gradient för positiva värden men kan ge döda neuroner vid negativa inputs.
**Leaky ReLU** — ReLU-variant som tillåter liten negativ slope för att undvika döda neuroner. Liten lutning (t.ex. 0,01) håller gradienten levande även när input är negativ.
**GELU** — Gaussian Error Linear Unit; mjuk gating använd i transformers, approximerar x·Φ(x). Mjukare än ReLU; standard i BERT, GPT och de flesta moderna språkmodeller.
**SiLU / Swish** — x·σ(x); mjuk icke-linearitet som ofta presterar bättre än ReLU i djupa nätverk. Själv-gated aktivering som Google introducerade; smooth derivata gynnar optimering.
**Sigmoid** — S-formad aktivering som mappar till (0,1); används historiskt och i gating. Lider av vanishing gradient i djupa nätverk; kvar i LSTM-gates och binär output.
**Tanh** — Hyperbolisk tangens; centrerad sigmoid med output i (-1,1). Noll-centrerad output kan accelerera konvergens jämfört med sigmoid i dolda lager.
**Softmax** — Normaliserar logits till sannolikhetsfördelning över klasser. Differentierbar och används i multi-klass klassificering och språkmodellers output-lager.
**Logits** — Råa osignerade scores före softmax eller sigmoid. Lineära projektioner av dolda tillstånd; temperature scaling och logit bias modifierar dem vid inferens.
**Weight Initialization** — Startvärden för vikter som påverkar konvergens och gradientstabilitet. Felaktig init kan ge exploding/vanishing gradients eller långsam konvergens redan från start.
**Xavier Initialization** — Varians skalad för att bevara aktiveringsvarians genom lager (tanh/sigmoid). Glorot & Bengio; balanserar fan-in och fan-out för symmetriska aktiveringar.
**He Initialization** — Varians skalad med fan-in för ReLU-nätverk. Kompenserar för ReLU:s halvering av varians; standard för CNN:er och ReLU-baserade arkitekturer.
**Backpropagation** — Effektiv beräkning av gradienter via kedjeregeln genom beräkningsgrafen. Gör träning av djupa nätverk praktiskt genom att undvika numerisk gradientberäkning per parameter.
**Computational Graph** — DAG som representerar operationer och möjliggör automatisk differentiering. Varje nod lagrar forward-värden; backward pass propagerar gradienter bakåt genom grafen.
**Automatic Differentiation** — Exakt gradientberäkning via symbolisk/registrerad graf snarare än numerisk approximation. Reverse-mode (backprop) är effektivt när antal parametrar >> antal outputs.
**Forward Pass** — Beräkning av outputs genom nätverket givet indata. Aktiveringar lagras för backward pass; minneskrävande vid långa sekvenser och stora modeller.
**Backward Pass** — Gradientberäkning från förlust tillbaka genom lagren. Kedjeregeln multiplicerar lokala gradienter; vanishing/exploding gradients uppstår vid långa kedjor.
**Gradient** — Partiell derivata av förlust w.r.t. parametrar; styr uppdateringsriktning. Storlek och riktning avgör hur varje vikt justeras för att minska förlusten.
**Gradient Descent** — Iterativ parameteruppdatering i negativ gradientriktning. Batch GD använder hela datasetet; praktiskt ersatt av stochastic/minibatch varianter.
**Stochastic Gradient Descent** — SGD med gradient estimerad från minibatch istället för hela datasetet. Introducerar brus som kan hjälpa generalisering och möjliggör online-lärande.
**Minibatch** — Delmängd av träningsexempel per uppdateringssteg; balanserar brus och effektivitet. GPU-parallellisering kräver batchar; storlek påverkar minne, brus och konvergens.
**Batch Size** — Antal exempel per gradientsteg; påverkar minne, brus och generalisering. Stora batchar ger stabilare gradienter men kan generalisera sämre; små batchar har mer brus men ofta bättre generalisering.
**Learning Rate** — Stegstorlek i parameteruppdatering; kritisk hyperparameter för konvergens. För hög LR ger instabilitet; för låg LR ger långsam konvergens och risk att fastna i platåer.
**Learning Rate Schedule** — Tidsvarierande learning rate (cosine, step decay, warmup) för stabil träning. Minskar LR mot slutet för finare konvergens; warmup stabiliserar tidiga transformer-träningssteg.
**Warmup** — Gradvis ökning av learning rate i början av träning för att undvika instabilitet. Särskilt viktigt för Adam och transformer-pretraining där initiala gradienter kan vara stora.
**Cosine Annealing** — Learning rate som följer cosinuskurva mot minimum. Ger mjuk avtagning utan abrupta steg; populärt schema i modern deep learning-träning.
**Momentum** — Ackumulerar velocity i gradientriktning för snabbare och stabilare konvergens. Dämpar oscillationer i ravine-liknande loss-landskap; klassisk förbättring av SGD.
**Nesterov Momentum** — Momentum som tittar framåt; ofta bättre konvergens än klassisk momentum. Utvärderar gradient vid extrapolerad position istället för nuvarande, ger bättre korrektion.
**Adam** — Adaptiv optimizer som kombinerar momentum och per-parameter learning rates. Uppskattar första och andra moment av gradienter; standardval men kan generalisera sämre än SGD med momentum i vissa fall.
**AdamW** — Adam med korrekt decoupled weight decay; standard i transformer-träning. Separerar weight decay från gradientuppdatering till skillnad från felaktig L2 i original Adam.
**RMSprop** — Adaptiv optimizer som skalar learning rate med rullande gradientkvadratmedel. Hinton; hanterar icke-stationära problem och RNN-träning där Adagrad minskar LR för snabbt.
**Adagrad** — Adaptiv optimizer med ackumulerad gradientkvadrat; learning rate minskar över tid. Bra för sparse features men LR kan bli för liten och stoppa inlärning helt.
**Lion** — Memory-efficient optimizer som använder tecken av momentum för uppdatering. Färre moment-estimat än Adam; lovande resultat med lägre minneskrav vid LLM-träning.
**Weight Decay** — L2-regularisering som krymper vikter vid varje steg. Motverkar overfitting genom att straffa stora viktstorlekar; i AdamW appliceras decoupled från gradientsteget.
**Gradient Clipping** — Begränsar gradientnorm för att förhindra explosiva uppdateringar. Global norm clipping är standard i RNN- och transformer-träning vid instabila loss-spikes.
**Vanishing Gradient** — Gradienter som exponentiellt krymper i djupa nätverk och hindrar inlärning i tidiga lager. Plågade sigmoid/tanh-RNN:er; mitigeras med ReLU, residual connections och normalisering.
**Exploding Gradient** — Gradienter som växer okontrollerat; kan orsaka numerisk instabilitet. Gradient clipping och korrekt initialisering är primära motmedel; vanligt i RNN:er med långa sekvenser.
**Batch Normalization** — Normaliserar aktiveringar per batch; stabiliserar träning och tillåter högre learning rates. Ioffe & Szegedy; minskar internal covariate shift men beteende skiljer sig mellan träning och inferens.
**Layer Normalization** — Normaliserar över feature-dimension per exempel; standard i transformers. Oberoende av batch size; fungerar väl med variabel sekvenslängd och små batchar.
**Group Normalization** — Normaliserar kanalgrupper; robust när batch size är liten. Alternativ till batch norm vid object detection och video där batch size ofta är 1–2.
**Instance Normalization** — Normaliserar per kanal och spatial position; vanligt i style transfer. Tar bort instance-specifik statistik och bevarar endast relativ struktur.
**RMSNorm** — Root Mean Square normalization; förenklad variant utan mean-centering, använd i LLaMA m.fl. Snabbare än layer norm med jämförbar stabilitet i stora språkmodeller.
**Dropout** — Slumpmässigt nollställer neuroner under träning som implicit ensemble. Srivastava et al.; approximerar medelvärde över delnätverk vid inferens via skalning.
**DropConnect** — Slumpmässigt nollställer vikter istället för neuroner. Generalisering av dropout; mindre vanligt men kan ge annorlunda regulariseringseffekt.
**Stochastic Depth** — Slumpmässigt hoppar över residual block under träning. Regulariserar djupa nätverk och minskar effektiv djup under träning; alla lager används vid inferens.
**Residual Connection** — Skip connection som adderar input till output: y = F(x) + x; möjliggör djupa nätverk. He et al.; löser degraderingsproblemet och underlättar gradientflöde genom identitetsvägar.
**Highway Network** — Gated skip connections som lär sig hur mycket transformation vs identitet som ska passera. Föregångare till ResNet; transform-gates styr informationsflöde per lager.
**DenseNet** — Varje lager tar input från alla föregående lager; stark feature-återanvändning. Huang et al.; effektiv parameteranvändning men högre minneskrav vid träning.
**Convolutional Layer** — Tillämpar lärd filter över lokala receptive fields för att fånga spatiala mönster. Parameter-delning och translation equivariance gör CNN:er effektiva för bilder och spatial data.
**Kernel / Filter** — Liten viktmatris som convolveras över indata. Flera filter lär olika detektors (kanter, texturer); djupare lager kombinerar lågnivå till högnivå features.
**Stride** — Stegstorlek då filtret förflyttas; större stride minskar spatial upplösning. Downsampling utan separat pooling; påverkar receptive field och beräkningskostnad.
**Padding** — Tillägg av border-värden för att kontrollera output-storlek. Same padding bevarar spatial dimension; valid padding minskar storlek vid varje lager.
**Receptive Field** — Region i indata som påverkar en given neurons aktivering. Växer med djup och filterstorlek; avgör hur stor kontext varje neuron kan se.
**Pooling Layer** — Aggregerar spatial information (max/avg) för dimensionsreduktion och translation invariance. Minskar spatial upplösning och beräkningskostnad; max pooling bevarar starkaste aktivering.
**Max Pooling** — Tar maximum inom varje poolingsfönster. Ger translation invariance och dimension reduction; kan förlora fin spatial information.
**Average Pooling** — Tar medelvärde inom poolingsfönster. Mjukare än max pooling; används i vissa arkitekturer och som global pooling före klassificeringshuvud.
**Global Average Pooling** — Medelvärde över hela spatial dimension; ersätter ofta FC-lager i CNNs. Lin et al.; minskar parametrar och minskar overfitting-risk i klassificeringshuvud.
**Transposed Convolution** — Upsampling-lager som lär sig spatial uppskalning; används i decoders och GANs. Kan producera checkerboard-artefakter; kräver noggrann kernel/stride-design.
**Dilated Convolution** — Convolution med hoppade kernel-positioner för större receptive field utan mer parametrar. Yu & Koltun; viktigt i semantic segmentation och tidsupplöst modellering.
**Depthwise Separable Convolution** — Faktoriserar convolution i depthwise + pointwise; effektivare i mobilarkitekturer. MobileNet; drastiskt färre parametrar och FLOPs med marginell accuracy-förlust.
**Recurrent Neural Network** — Nätverk med feedback-loopar för sekvensmodellering. Processar sekvenser steg för steg med dolt tillstånd; ersatt av transformers för de flesta NLP-uppgifter.
**LSTM** — Long Short-Term Memory; gated RNN som hanterar långsiktiga beroenden. Input, forget och output gates styr informationsflöde; mitigerar vanishing gradient bättre än vanilla RNN.
**GRU** — Gated Recurrent Unit; förenklad LSTM-variant med färre parametrar. Kombinerar forget och input gate; ofta jämförbar prestanda med färre beräkningar.
**Bidirectional RNN** — Processar sekvens i båda riktningar; kräver hel sekvens tillgänglig. Ger rikare kontext per position men dubblerar beräkning och minne.
**Seq2Seq** — Encoder-decoder-arkitektur som mappar en sekvens till en annan. Grund för tidig maskinöversättning; encoder komprimerar input, decoder genererar output autoregressivt.
**Teacher Forcing** — Under träning matas decoder med ground truth tokens istället för egna prediktioner. Accelererar träning men skapar exposure bias vid inferens.
**Exposure Bias** — Mismatch mellan träning (teacher forcing) och inferens (autoregressiv sampling). Modellen tränas på korrekta prefix men genererar från egna (potentiellt felaktiga) tokens.
**Attention Mechanism** — Dynamisk viktning av relevanta delar av input; kärnan i transformers. Ersatte fixed-size bottleneck i seq2seq och möjliggör långa beroenden.
**Self-Attention** — Attention där queries, keys och values kommer från samma sekvens. Varje token kan direkt attenda över alla andra; O(n²) komplexitet i sekvenslängd.
**Multi-Head Attention** — Parallella attention-huvuden som fångar olika relationstyper. Varje huvud projicerar till delrum; aggregering ger rikare representationer.
**Positional Encoding** — Injicerar positionsinformation i sekvensrepresentationer. Transformers saknar inbyggd ordning; position måste kodas explicit eller via strukturell bias.
**Sinusoidal Positional Encoding** — Fast sin/cos-kodning av position från original transformer-papperet. Generaliserar delvis till längre sekvenser; kräver ingen extra träning.
**Learned Positional Embedding** — Tränade positionvektorer istället för fast encoding. Enkelt men begränsat av max-sekvenslängd vid träning; extrapolation kräver extra tekniker.
**Rotary Position Embedding** — RoPE; roterar query/key-vektorer för relativ positionskodning. Su et al.; naturlig relativ encoding som dominerar moderna LLM:er.
**ALiBi** — Attention with Linear Biases; extrapolerar till längre sekvenser utan explicit positionsembedding. Press et al.; avståndsbaserade bias i attention-scores.
**Flash Attention** — IO-aware exakt attention-algoritm som minskar HBM-åtkomst och minneskrav. Dao et al.; blockvis beräkning som undviker materialisering av full n×n-matris.
**Mixed Precision Training** — Träning med FP16/BF16 för hastighet kombinerat med FP32 master weights. Tensor Cores accelererar matmul; loss scaling hanterar FP16:s smala dynamiskt område.
**BF16** — Brain Float 16; bredare exponent än FP16, ofta mer stabilt för deep learning. Samma exponentområde som FP32; standard på TPU och många GPU-träningsjobb.
**FP16** — 16-bitars flyttal; halverar minne men risk för underflow/overflow. Kräver loss scaling och master weights i FP32; snabbare på hårdvara med Tensor Core-stöd.
**TF32** — NVIDIA-format som accelererar matmul med reducerad precision internt. Ampere+ GPU:er; transparent acceleration utan explicit kodändring i de flesta ramverk.
**Gradient Accumulation** — Ackumulerar gradienter över flera microbatches för effektiv större batch size. Simulerar stor batch på begränsat GPU-minne; synkroniserar uppdatering efter N steg.
**Gradient Checkpointing** — Rekomputera activations vid backward pass för att spara minne. Chen et al.; tradeoff minne mot extra forward-beräkning; möjliggör träning av större modeller.
**Knowledge Distillation** — Mindre student-modell lär sig matcha större teacher-modells outputs eller logits. Hinton et al.; komprimerar kunskap till mindre modell med mjukare träningsmål.
**Model Compression** — Tekniker (pruning, quantization, distillation) för mindre/snabbare modeller. Kritiskt för edge deployment och kostnadseffektiv inferens i produktion.
**Pruning** — Borttagning av vikter/neuroner med liten påverkan på output. Strukturerad pruning tar hela kanaler; ostrukturerad ger högre komprimering men kräver sparse hårdvara.
**Quantization** — Reducerar numerisk precision (INT8/INT4) för inferens eller träning. Post-training ger snabb komprimering; QAT ger bättre noggrannhet vid låg precision.
**Post-training Quantization** — Kvantisering efter träning utan finjustering. Snabbast att applicera men kan ge större accuracy-förlust än quantization-aware training.
**Quantization-aware Training** — Simulerar kvantisering under träning för bättre INT-precision. Modellen lär sig kompensera för kvantiseringsfel; rekommenderas för INT4/INT8 deployment.
**LoRA** — Low-Rank Adaptation; tränar lågrangsuppdateringar av vikter istället för full fine-tuning. Hu et al.; dramatiskt färre träningsparametrar med jämförbar downstream-prestanda.
**Parameter-efficient Fine-tuning** — PEFT; familj av metoder som uppdaterar liten delmängd av parametrar. Inkluderar LoRA, adapters, prefix tuning; möjliggör finjustering av stora modeller på begränsad hårdvara.
**Neural Architecture Search** — Automatiserad sökning efter optimal nätverksarkitektur. Zoph & Le; dyrt i compute men har producerat EfficienNet och andra SOTA-arkitekturer.
**Universal Approximation Theorem** — Teoretiskt resultat: tillräckligt brett MLP kan approximera kontinuerliga funktioner. Garanti om existens, inte om inlärbarhet eller sample efficiency.
**Mode Collapse** — Generativ modell producerar begränsad variation av outputs. Vanligt i GANs där generatorn hittar få modes som foolar diskriminatorn.
**Dead ReLU** — Neuron vars vikt aldrig uppdateras eftersom input alltid är negativ. Leaky ReLU och proper init mitigerar; kan reducera effektiv kapacitet i nätverket.
**Spectral Normalization** — Begränsar singularvärden av vikter för Lipschitz-stabilitet i GANs. Miyato et al.; stabiliserar GAN-träning genom att begränsa diskriminatorns Lipschitz-konstant.
**Early Stopping** — Avbryter träning när valideringsförlust slutar förbättras; implicit regularisering. Förhindrar overfitting utan explicit penalty; enklaste regulariseringstekniken i praktiken.

## Transformers & Attention

**Self-Attention** — Attention där query, key och value hämtas från samma sekvens; möjliggör parallell kontextuell modellering utan rekurrens. Varje token beräknar relevans mot alla andra i O(n²) tid och minne.
**Cross-Attention** — Attention där queries kommer från en sekvens och keys/values från en annan; central i encoder-decoder och multimodala modeller. Kopplar t.ex. decoder till encoder-representationer eller text till bildfeatures.
**Scaled Dot-Product Attention** — Beräknar softmax(QK^T/√d_k)V; skalningen stabiliserar gradienter när dimensionsstorleken växer. Vaswani et al.; standard attention-formulering i transformers.
**Multi-Head Attention** — Parallella attention-huvuden projicerar Q/K/V till delrum och aggregerar flera relationstyper. Varje huvud kan fokusera på olika syntaktiska eller semantiska relationer.
**Query-Key-Value Projection** — Linjära projektioner som mappar dolda tillstånd till Q, K och V för attention-beräkning. Separata viktmatriser W_Q, W_K, W_V per huvud; tränas end-to-end.
**Causal Mask** — Triangulär mask som förhindrar att positioner ser framtida tokens; krävs för autoregressiv generering. Säkerställer att prediktion vid position t endast conditionar på tokens 0..t-1.
**Bidirectional Attention** — Fullständig token-till-token-interaktion utan kausal begränsning; används i encoders som BERT. Varje token ser hela sekvensen; olämpligt för autoregressiv generering utan maskering.
**Encoder-Decoder Attention** — Decoder-queries attendar över encoder-keys/values; kopplar käll- och målsekvenser. Central i original transformer för översättning; cross-attention-lager efter self-attention i decoder.
**Pre-Norm Transformer** — Layer normalization placeras före attention/FFN; ofta stabilare träning i djupa modeller. Xiong et al.; gradientflöde genom residual paths förbättras jämfört med post-norm.
**Post-Norm Transformer** — Original transformer-layout med normalisering efter sub-lager; kräver ofta warmup. Norm efter residual add; kan vara instabil i mycket djupa modeller utan noggrann LR-schedulering.
**Feed-Forward Network Block** — Position-wise två-lagers MLP med icke-linearitet mellan attention-lager; ~2/3 av transformer-parametrar. Expanderar typiskt dimension 4× (d → 4d → d); samma MLP per token-position.
**Position-Wise FFN** — Samma MLP tillämpas oberoende på varje token-position utan parameterdelning över sekvensen. Till skillnad från convolutions delas inga parametrar mellan positioner i FFN-blocket.
**Sinusoidal Positional Encoding** — Fast sin/cos-kodning av absolut position; generaliserar delvis till längre sekvenser än träningslängd. Vaswani et al.; olika frekvenser per dimension kodar position unikt.
**Learned Positional Embedding** — Tränade positionvektorer adderas till token-embedding; enkelt men begränsat av max-sekvenslängd. Kräver omträning eller interpolation för längre sekvenser vid inferens.
**Rotary Position Embedding** — RoPE roterar Q/K i komplexplan baserat på position; kodar relativ position naturligt. Su et al.; de facto standard i LLaMA, Mistral och de flesta moderna LLM:er.
**Relative Position Bias** — Additiv bias till attention-scores baserad på tokenavstånd; fångar lokal struktur utan absolut embedding. T5 och BERT-varianter; lär sig avståndsprior direkt i attention.
**ALiBi** — Attention with Linear Biases: avståndsbaserade straff i attention utan explicit positionsembedding. Press et al.; enkel extrapolation till längre kontext utan extra positionsträning.
**Flash Attention** — IO-aware blockvis attention som minimerar HBM-läs/skriv; exakt men betydligt snabbare och minnessnålare. Dao et al.; undviker materialisering av n×n attention-matris i HBM.
**FlashAttention-2** — Förbättrad parallellisering och arbetsfördelning i Flash Attention; högre GPU-utnyttjande. Dao; bättre work partitioning och reducerad synkronisering mellan warps.
**PagedAttention** — Virtuell minneshantering för KV-cache i block; möjliggör effektiv batched inferens med varierande sekvenslängder. vLLM; inspirerad av OS paging; minskar minnesfragmentering vid serving.
**KV Cache** — Cachade key/value-tensorer från tidigare tokens vid autoregressiv inferens; undviker omräkning. Dominerande minneskostnad vid lång generering; MQA/GQA minskar cache-storlek.
**Multi-Query Attention** — MQA: delade K/V-projektioner över huvuden; minskar KV-cache-minne vid inferens. Shazeer; marginell kvalitetsförlust men dramatiskt mindre cache för lång generering.
**Grouped-Query Attention** — GQA: mellanting mellan MHA och MQA; grupper av huvuden delar K/V för minnes-/kvalitetsbalans. Ainslie et al.; används i LLaMA 2/3 som kompromiss mellan kvalitet och inferenshastighet.
**Sliding Window Attention** — Varje token attendar endast inom lokalt fönster; linjärt minne i sekvenslängd. Mistral m.fl.; begränsar långdistansberoenden men skalar till mycket långa sekvenser.
**Longformer Attention** — Kombinerar lokalt fönster med få globala tokens för långkontext utan full kvadratisk kostnad. Beltagy et al.; globala tokens agerar hubbar för långdistansinformation.
**BigBird Attention** — Sparse mönster med random, window och global tokens; teoretisk universal-approximator-garanti. Zaheer et al.; O(n) komplexitet med bevisad expressivitet under vissa antaganden.
**Linformer Attention** — Lågranksaproximation av attention-matris via projicerade keys; O(n) i sekvenslängd. Wang et al.; projicerar K/V till lägre dimension; approximativ men skalbar.
**Performer Attention** — Approximerar softmax-kernel med FAVOR+ för linjärt minne och tid. Choromanski et al.; random feature approximation; exakt attention approximeras med kontrollerbart fel.
**Linear Attention** — Ersätter softmax med kernel-trick som tillåter associativ beräkning; sub-kvadratisk. Katharopoulos et al.; O(n) tid och minne men ofta lägre kvalitet än softmax attention.
**Sparse Attention Pattern** — Fördefinierat gles attention-mönster som minskar beräkningskostnad för långa sekvenser. Strided, fixed eller learned patterns; tradeoff mellan expressivitet och effektivitet.
**Local Attention** — Begränsad receptive field per token; används i bild-transformers och långtext. Varje position ser endast närliggande tokens; hierarkiska modeller stackar för global kontext.
**Global Attention Tokens** — Utvalda tokens med full sekvensåtkomst som agerar informationshubbar. Longformer och BigBird; få globala tokens ger långdistans-koppling utan full O(n²) kostnad.
**Attention Dropout** — Slumpmässig nollställning av attention-vikter efter softmax; regularisering under träning. Appliceras på attention-probabilities; minskar co-adaptation mellan huvuden.
**Attention Sink** — Tendens att första token får oproportionerligt hög attention-vikt; relevant vid prompt-cache. Xiao et al.; kan utnyttjas för effektiv streaming inferens med cached prefix.
**Induction Head** — Attention-mechansim som kopierar mönster från tidigare kontext; kopplat till in-context learning. Olsson et al.; två-lagers circuit som matchar och reproducerar sekvensmönster.
**Transformer Block** — Standard enhet: attention + residual + FFN + residual, eventuellt med normalisering. Upprepas N gånger; djup och bredd (d_model, n_heads) styr kapacitet och compute.
**Encoder-Only Transformer** — Stack av self-attention-block utan kausal mask; för förståelseuppgifter. BERT, RoBERTa; bi-directional kontext för klassificering, NER och embedding.
**Decoder-Only Transformer** — Kausalt maskerad stack; grund för moderna LLM:er. GPT, LLaMA; autoregressiv nästa-token-prediktion med causal self-attention.
**Encoder-Decoder Transformer** — Separata encoder och decoder med cross-attention; seq2seq och översättning. T5, BART; encoder processar input, decoder genererar output autoregressivt.
**Token Mixing** — Kombination av attention (tokenmixning) och FFN (kanalmixning) i vision transformers. Attention blandar information mellan spatiala positioner; FFN processar per-token features.
**Patch Embedding** — Delar upp bild i patchar som linjärt projiceras till token-vektorer; ViT-ingång. Dosovitskiy et al.; behandlar bildpatches som ord i en sekvens för transformer.
**CLS Token** — Special token vars representation aggregerar sekvensinformation för klassificering. BERT; prepended token som samlar global kontext via self-attention.
**Sequence Packing** — Packar flera korta sekvenser i en batch-rad med separata attention-masker; högre GPU-utnyttjande. Undviker padding-waste; kräver noggrann maskering för att förhindra cross-sequence attention.
**Context Parallelism** — Delar lång sekvens över enheter med kommunikation vid attention-gränser. Ring attention m.fl.; möjliggör träning och inferens med sekvenser som inte får plats på en GPU.
**Tensor Parallelism in Attention** — Delar QKV-projektioner och attention-beräkning horisontellt över GPU:er. Megatron-LM; all-reduce vid attention-output; nödvändigt för modeller som inte får plats på en enhet.
**Ring Attention** — Distribuerar KV-block i ring-topologi för att träna extremt långa sekvenser. Blockvis attention med kommunikation längs ring; minimerar peak minne per enhet.
**NTK-Aware Scaling** — Skalar RoPE-bas för att interpolera till längre kontext utan finjustering. Bloc et al.; justerar frekvensbas så att positioner mappas till träningsintervall vid extrapolation.
**YaRN** — Yet another RoPE extensioN: finjusterar frekvensinterpolering för längre kontext med minimal degradering. Peng et al.; kombinerar NTK-scaling med finmaskig frekvensjustering.
**Position Interpolation** — Komprimerar position-index vid inferens för att passa tränad max-längd; enkel kontextförlängning. Chen et al.; linjär skalning av positioner till träningsintervall.
**Extrapolation vs Interpolation** — Extrapolation testar längre sekvenser direkt; interpolation mappar positioner till träningsintervall. Extrapolation ger ofta sämre kvalitet; interpolation är enklare men kan sudda ut fin positionsskillnad.
**Attention Temperature** — Skalning av QK^T före softmax; högre temperatur ger mjukare fördelning. √d_k är standard; högre temperatur sprider attention; lägre ger skarpare fokus.
**Softmax Numerical Stability** — Subtraherar max(QK^T) före exp för att undvika overflow i attention. Standard trick i alla softmax-implementationer; påverkar inte resultat men förhindrar NaN.
**Disentangled Attention** — DeBERTa: separerar innehålls- och positionsrepresentationer i Q/K. He et al.; förbättrar förståelseuppgifter genom explicit positions- och innehållsinteraktion.
**Talking-Heads Attention** — Extra linjära transformationer före/efter softmax över huvuden. Shazeer et al.; låter huvuden interagera före och efter attention-viktning.
**Multi-Scale Attention** — Kombinerar attention över flera upplösningar eller fönsterstorlekar. Fångar både lokal och global struktur; vanligt i vision transformers och långtext-modeller.
**Cross-Layer Attention** — Attention mellan representationer från olika lager; sällsynt men används i vissa arkitekturer. Tillåter direkt informationsflöde mellan icke-intilliggande lager.
**Memory-Augmented Transformer** — Externa minnesmoduler som tokens kan attenda över utöver aktuell sekvens. Differentiable neural computer-liknande; utökar effektiv kontext utan längre sekvens.
**Transformer-XL Segment Recurrence** — Cachar dolda tillstånd från föregående segment för längre beroenden. Dai et al.; relativ position inom segment; överlappande segment ger kontinuerlig kontext.
**Compressive Transformer** — Komprimerar äldre minne till sammanfattningsvektorer för längre effektiv kontext. Rae et al.; hierarkiskt minne med komprimerade representationer av äldre tokens.
**Perceiver IO** — Latent bottleneck som attendar över indata av godtycklig storlek/form; generisk multimodal encoder. Jaegle et al.; fix antal latenta queries komprimerar variabel input.
**Perceiver Resampler** — Fast antal latenta vektorer som komprimerar variabel indata till fix representation. Används i Flamingo m.fl. för att mappa bild/video till LLM-kompatibel representation.
**Attention Map Visualization** — Heatmap över attention-vikter; diagnostiskt men tolkning kräver försiktighet. Attention ≠ förklaring; huvuden kan ha olika semantik; visualisering är heuristisk.
**Head Pruning** — Borttagning av attention-huvuden med litet bidrag till downstream-prestanda. Michel et al.; upp till 40% av huvuden kan tas bort med minimal accuracy-förlust i vissa modeller.
**Attention Entropy** — Entropi av attention-fördelning; låg entropi indikerar skarp fokusering. Hög entropi = diffus attention över sekvensen; kan indikera osäkerhet eller bred kontext.
**Causal Language Modeling Head** — Linjärt lager + softmax som predicerar nästa token från dolda tillstånd. Standard output för GPT-liknande modeller; delade eller separata embedding-vikter.
**Masked Language Modeling Head** — Predicerar maskerade tokens från bidirectional kontext; BERT-style. Endast maskerade positioner bidrar till loss; tränar rik kontextuell representation.
**Attention as Soft Dictionary Lookup** — Tolkning: softmax-viktad summa av values som lookup via key-matchning. Query söker i key-space; viktad summa av values ger output; intuitiv men förenklad bild.
**Low-Rank Attention Approximation** — Faktoriserar attention-matris för att minska beräkning och minne. Linformer, Performer m.fl.; approximerar full attention med kontrollerbart fel.
**Blockwise Parallel Attention** — Beräknar attention i block parallellt; grund för Flash Attention. Tile-baserad beräkning som minimerar HBM-traffic; online softmax över block.
**Online Softmax** — Inkrementell softmax-beräkning över block utan full materialisering av score-matris. Flash Attention; uppdaterar running max och sum exp per block.
**Stochastic Attention** — Slumpmässigt subsampling av keys/values under träning som regularisering. Zheng et al.; approximerar full attention med brus som kan förbättra generalisering.
**Cross-Document Attention** — Attention över flera dokument i samma sekvens; kräver noggrann maskering. Long-context RAG; dokument-gränser måste respekteras i attention-mask.
**Document-Level Attention Mask** — Separerar attention inom dokument för att undvika otillåten cross-doc-leakage. Varje dokument får egen attention-submatris; förhindrar att modellen blandas mellan källor.
**Prefix Attention Mask** — Tillåter full bidirectional attention inom prompt-prefix och kausal generering efter. Prefix-LM; används i vissa finjusterings- och inferensscenarier.
**U-Net Transformer Hybrid** — Kombinerar U-Net-skip med transformer-block i diffusion och vision. Skip connections mellan encoder/decoder-nivåer; multi-scale features i generativa modeller.
**Swin Shifted Window** — Shifted window attention i hierarkisk vision transformer för cross-window-koppling. Liu et al.; alternerande window shifts ger cross-window kommunikation utan global O(n²).
**Axial Attention** — Factoriserar 2D-attention till rad- och kolumnpass; effektiv för bilder. Ho et al.; O(n√n) istället för O(n²) för n×n spatial grid.
**Factorized Attention** — Delar upp full attention i faktoriserade steg över dimensioner eller positioner. Reducerar komplexitet genom sekventiella attention-pass istället för full n×n.
**Token Dropping in Attention** — Dynamiskt hoppar över tokens med låg attention-saliency för acceleration. Adaptive computation; sparar compute på mindre viktiga positioner.
**Query-Key Normalization** — QK-norm stabiliserar attention i stora modeller genom att normalisera projektioner. Henry et al.; förhindrar attention-logit-explosion i djupa transformers.
**Attention Logit Capping** — Begränsar max attention-score för att förhindra dominans av enskilda keys. Gemma m.fl.; maxcap på logits före softmax ger mer jämn attention-fördelning.
**Relative Attention Core** — Generellt ramverk för att injicera strukturell prior via relativ position/innehåll. Enar relativ position bias, RoPE och ALiBi under gemensamt perspektiv.
**Transformer Depth vs Width** — Tradeoff mellan antal lager och dold dimension; påverkar kapacitet och parallellism. Djupare modeller lär hierarkier; bredare ger mer parallell kapacitet per lager.
**Attention FLOPs Scaling** — Attention-kostnad skalar O(n²·d) i sekvenslängd n och dimension d. Dominerar compute vid långa sekvenser; motiv för sparse/linear attention-forskning.
**Memory Bandwidth Bottleneck** — Attention ofta minnesbandbreddsbegränsad snarare än compute-bound på GPU. Flash Attention adresserar detta genom att minska HBM-åtkomst; IO-aware algoritmer kritiska.
**Speculative Decoding with KV Reuse** — Draft-modell delar eller approximerar KV-cache för snabbare validering. Minskar redundant beräkning mellan draft och target modell vid spekulativ decoding.
**Multi-Token Prediction Head** — Auxiliary heads som predicerar flera framtida tokens parallellt; accelererar träning. Gloeckle et al.; extra supervision signal; kan också användas vid inferens (Medusa).
**Mixture-of-Depths** — Dynamiskt hoppar över lager per token baserat på router; sparar compute. Raposo et al.; conditional computation; inte alla tokens behöver alla lager.
**LayerDrop** — Slumpmässigt droppar hela transformer-lager under träning som regularisering. Fan et al.; approximerar ensemble över sub-nätverk; alla lager används vid inferens.
**Sandwich Normalization** — Extra normalisering runt sub-lager; vissa arkitekturer för stabil deep training. Norm både före och efter sub-lager; experimentellt för mycket djupa modeller.

## Stora språkmodeller

**Large Language Model** — Skalad autoregressiv transformer tränad på massiv textkorpus; generaliserar brett via nästa-token-prediktion. Kapacitet och emergenta förmågor växer med modellstorlek, data och compute enligt scaling laws.
**Autoregressive Generation** — Sekventiell token-generering där varje steg conditionar på alla tidigare tokens. Standard för GPT-liknande modeller; O(n) steg för n genererade tokens med KV-cache-optimering.
**Next-Token Prediction** — Träningsmål: maximera sannolikhet för korrekt nästa token givet prefix. Enkel objektivfunktion som indirekt lär grammatik, fakta och resonemang från textmassa.
**Tokenization** — Mappning text→diskreta token-ID via BPE, WordPiece eller SentencePiece. Subword-tokenisering balanserar vokabulärstorlek och sekvenslängd; påverkar alla downstream-uppgifter.
**Byte-Pair Encoding** — BPE: iterativt slår samman frekventa bytepar till subword-vokabulär. Sennrich et al.; börjar med bytes/tecken och bygger vokabulär från frekvensstatistik.
**SentencePiece** — Language-agnostic tokenisering direkt på rå bytes/Unicode utan förbehandling. Kudo & Richardson; hanterar whitespace och språk med delat vokabulär; används i T5 och LLaMA.
**Vocabulary Size** — Antal unika tokens; större vokabulär minskar sekvenslängd men ökar embedding-parametrar. Typiskt 32k–128k för moderna LLM:er; tradeoff mellan sekvenslängd och embedding-minne.
**Context Window** — Max antal tokens modellen kan processa i en forward pass; begränsar prompt+generering. Växer från 2k till 128k+ via arkitektur (RoPE, ALiBi) och träning på långa sekvenser.
**Long-Context LLM** — Modell optimerad för kontext >> träning via arkitektur, data eller finjustering. Position interpolation, YaRN och lång-sekvens träning möjliggör 100k+ tokens kontext.
**Emergent Abilities** — Förmågor som dyker upp vid skalning och inte var förutsägbara från mindre modeller. Wei et al.; t.ex. chain-of-thought, multi-step reasoning; debatteras om de är verkligt emergenta.
**In-Context Learning** — Modellen löser nya uppgifter via exempel i prompten utan gradientuppdatering. Brown et al.; few-shot via demonstrations; mekanism kopplas till induction heads och attention.
**Few-Shot Prompting** — Ger några demonstrations-exempel i prompten för att styra beteende. Ingen finjustering; kvalitet beror på exempelval, ordning och prompt-formulering.
**Zero-Shot Prompting** — Instruktion utan exempel; förlitar sig på förtränad kunskap och instruktionsföljning. Kräver modell tränad på instruktionsliknande data; instruction tuning förbättrar kraftigt.
**Chain-of-Thought** — CoT: modellen genererar mellansteg som förbättrar resonemang på komplexa frågor. Wei et al.; "think step by step"; dramatisk förbättring på matematik och logik.
**System Prompt** — Fast instruktion som sätter roll, ton och begränsningar för hela konversationen. Sätter beteende före användarmeddelanden; API:er som OpenAI och Anthropic exponerar detta explicit.
**Prompt Engineering** — Design av indataformulering för att maximera kvalitet utan modelländring. Inkluderar struktur, exempel, rollbeskrivning och output-format; billigare än finjustering men begränsad av modellkapacitet.
**Temperature Sampling** — Skalar logits före sampling; högre T ger mer variation, lägre T mer deterministiskt. T→0 approximerar greedy; T>1 ökar kreativitet men risk för incoherence.
**Top-k Sampling** — Begränsar sampling till k mest sannolika tokens vid varje steg. Fan et al.; förhindrar sampling av extremt osannolika tokens; k=40–50 vanligt i praktiken.
**Top-p Nucleus Sampling** — Sample från minsta tokenmängd vars kumulativa sannolikhet ≥ p. Holtzman et al.; adaptiv storlek på sampling-mängd; mer dynamiskt än fix top-k.
**Min-p Sampling** — Filtrerar tokens under dynamisk sannolikhetströskel relativt top-token. Nyare metod; kombinerar adaptiv filtrering med relativ tröskel för bättre kvalitet vid låg temperatur.
**Repetition Penalty** — Straffar logits för nyligen genererade tokens för att minska loopar. Multiplicativ eller additiv penalty på redan genererade tokens; viktigt för öppen-ended generering.
**Frequency Penalty** — Linjärt straff per token baserat på historisk frekvens i genererad text. OpenAI API-parameter; minskar repetitiva fraser proportionellt mot antal förekomster.
**Presence Penalty** — Engångsstraff om token redan förekommit; uppmuntrar nytt innehåll. Straffar oavsett hur många gånger token förekommit; mer aggressiv diversifiering än frequency penalty.
**Beam Search** — Bevarar flera hypoteser parallellt; vanligt i maskinöversättning, sällan i chat-LLM. Deterministisk men kan ge repetitiv text; ersatt av sampling i öppen dialog.
**Greedy Decoding** — Väljer argmax-token varje steg; snabbt men ofta repetitivt och suboptimalt. Lokalt optimalt per steg men globalt suboptimalt; baseline för enkel inferens.
**Stop Sequence** — Special token eller sträng som avslutar generering när den produceras. EOS-token eller custom stop strings; förhindrar oändlig generering och kontrollerar output-längd.
**Max Tokens** — Övre gräns på genererade tokens; kontrollerar kostnad och latens. API-limit och användarkontroll; måste balanseras mot uppgiftens krav på svarslängd.
**Prefill Phase** — Initial forward pass som processar hela prompten och bygger KV-cache. Compute-bound och parallelliserbar; dominerar TTFT vid långa prompts.
**Decode Phase** — Autoregressiva steg som appendar en token i taget efter prefill. Memory-bandwidth-bound; KV-cache läses varje steg; tokens/s är nyckelmetrik.
**Time to First Token** — TTFT: latens från request till första genererade token; domineras av prefill. Kritiskt för interaktiv UX; prompt caching och disaggregated prefill minskar TTFT.
**Tokens Per Second** — Genomströmning i decode-fas; nyckelmetrik för inferensprestanda. Påverkas av modellstorlek, batch size, kvantisering och hårdvara; viktigt för serving-kostnad.
**Mixture of Experts LLM** — MoE-LLM: sparsam aktivering av expert-FFN per token för skalad kapacitet. Mixtral, GPT-4 (rykten); fler parametrar men liknande inferens-FLOPs som dense modell.
**Expert Routing** — Router väljer top-k experter per token; load balancing är träningsutmaning. Aux loss för jämn expert-användning; risk för expert collapse utan balansering.
**Dense vs MoE LLM** — Dense aktiverar alla parametrar; MoE aktiverar delmängd vid inferens. MoE ger mer kapacitet per FLOP men komplexare serving och all-to-all kommunikation.
**Chinchilla Optimal** — Empiriskt optimalt förhållande compute/data/modellstorlek enligt scaling laws. Hoffmann et al.; modeller tränades för litet data; ~20 tokens per parameter rekommenderas.
**Compute-Optimal Training** — Träningsbudget allokeras för att balansera modellstorlek och tokens enligt scaling law. Undviker underträning av stora modeller; Chinchilla omdefinierade LLM-träningsrecept.
**Data Contamination** — Benchmark-data i träningskorpus som inflates utvärderingsresultat. Kritiskt metodologiskt problem; kräver deduplicering och held-out benchmarks för opartisk utvärdering.
**Memorization vs Generalization** — LLM kan memorerar sekvenser vs lära generaliserbara mönster; svårt att skilja. Carlini et al.; memorization ökar med modellstorlek och upprepning i träningsdata.
**Hallucination** — Generering av plausible men faktiskt felaktigt innehåll med hög konfidens. Kvarstående problem trots skalning; RAG och verifiering mitigerar men eliminerar inte.
**Confabulation** — Synonym till hallucination; modellen fyller i luckor med fabricerade detaljer. Särskilt vid saknad kunskap eller överconfidence; skiljer sig från medveten fabricering.
**Instruction Tuning** — Finjustering på (instruktion, svar)-par för bättre användarlydnad. FLAN, Alpaca; förbättrar zero-shot uppgiftsföljning utan RLHF; ofta steg före alignment.
**Chat Template** — Formateringskonvention som wrappar meddelanden i special tokens för konversationsmodeller. Jinja2-mallar i HuggingFace; fel template ger degraderad kvalitet vid inferens.
**Base Model vs Instruct Model** — Base är rå förträning; instruct är finjusterad för dialog och uppgiftsföljning. Base kräver prompt engineering; instruct följer instruktioner direkt men kan vara mer restriktiv.
**Continued Pretraining** — Extra förträning på domänspecifik korpus före downstream fine-tuning. Anpassar modell till juridik, medicin, kod m.fl.; risk för catastrophic forgetting av generell kunskap.
**Mid-Training** — Mellansteg med curated data mellan massiv pretrain och alignment. Högkvalitativ syntetisk eller filtrerad data; förbättrar kapacitet före instruction tuning och RLHF.
**Synthetic Instruction Data** — AI-genererade instruktionssvar för att skala finjusteringsdata. Alpaca, Self-Instruct; kvalitetskontroll kritisk; modell-kollaps risk vid överreliance på syntetisk data.
**Self-Instruct** — Modell genererar egna instruktioner och svar som träningsdata. Wang et al.; bootstrapping av instruktionsdata utan mänsklig annotation; seed med få manuella exempel.
**Constitutional AI Data** — Självkritik och revision enligt principer för att producera alignment-data. Anthropic; modell kritiserar och förbättrar egna svar enligt konstitutionella principer.
**Token Healing** — Reparera tokengränser vid streaming så att ofullständiga UTF-8-sekvenser inte bryts. vLLM och andra servrar; viktigt för korrekt visning av streaming output.
**Logit Bias** — Additiv offset på specifika token-logits för att styra output under inferens. OpenAI API; kan tvinga eller förbjuda specifika tokens utan prompt-ändring.
**Structured Output Mode** — Tvingar JSON/schema via constrained decoding eller grammar guidance. Outlines, Guidance; garanterar syntaktiskt giltig output enligt schema.
**JSON Mode** — API-läge som instruerar modellen att producera giltig JSON. Post-processing och constrained decoding; viktigt för verktygsintegration och API-responser.
**Function Calling** — Modellen väljer verktyg och argument i strukturerat format för exekvering. OpenAI tools API; modellen genererar JSON med funktionsnamn och argument.
**Tool Use** — LLM delegerar beräkning, sökning eller API-anrop till externa verktyg. ReAct, function calling; utökar modellens kapacitet bortom parametrisk kunskap.
**Agentic Loop** — Iterativ cykel: planera → verktyg → observera → uppdatera tills mål uppnås. ReAct, AutoGPT; modellen agerar autonomt med verktyg och minne över flera steg.
**Reasoning Model** — Modell tränad/finjusterad för längre intern resonemang före svar (o1-liknande). OpenAI o1; process supervision och RL med verifierbar reward; mer compute vid inferens.
**Test-Time Compute Scaling** — Allokerar mer inferens/compute vid svar för bättre kvalitet (best-of-N, tree search). Snell et al.; mer tänkande vid inferens kan kompensera mindre modell.
**Best-of-N Sampling** — Genererar N kandidater och väljer bästa via verifierare eller reward model. Ökar kvalitet utan viktuppdatering; kostnad skalar linjärt med N.
**Process Reward Model** — PRM: betygsätter mellansteg i resonemang, inte bara slutresultat. Lightman et al.; bättre signal för multi-step reasoning än outcome-only reward.
**Outcome Reward Model** — ORM: betygsätter slutligt svar baserat på korrekthet eller preferens. Enklare att träna men svagare signal för resonemangssteg; kan belöna rätt svar med felaktigt resonemang.
**Model Merging** — Kombinerar vikter från flera finjusterade modeller (SLERP, TIES, DARE). MergeKit; skapar multi-capability modell utan multi-task träning; heuristik för konfliktlösning.
**Quantized LLM** — LLM i INT4/INT8/FP8 för minskat minne och snabbare inferens. GPTQ, AWQ, GGUF; marginell kvalitetsförlust vid 4-bit med rätt metod.
**GGUF Format** — Kvantiserat modellformat för lokal inferens med llama.cpp. Komprimerade vikter med metadata; populärt för att köra LLM lokalt på CPU/GPU.
**Speculative Decoding** — Liten draft-modell föreslår tokens som stor modell verifierar parallellt. Leviathan et al.; 2–3× speedup utan kvalitetsförlust om draft matchar target distribution.
**Medusa Heads** — Extra decoding-huvuden för parallell multi-token-prediktion vid inferens. Cai et al.; flera tokens per forward pass; kompletterar speculative decoding.
**Prompt Caching** — Återanvänder KV-cache för identiska prompt-prefix; sänker latens och kostnad. Anthropic, OpenAI; prefix-hashning; dramatisk besparing vid långa system prompts.
**Prefix Caching** — Server-side cache av beräknade prefix-representationer mellan requests. vLLM automatic prefix caching; delad cache i multi-tenant serving.
**Batch Inference** — Processar flera prompts parallellt med padding/packing för GPU-effektivitet. Högre throughput än sekventiell inferens; continuous batching optimerar ytterligare.
**Continuous Batching** — Dynamiskt lägger till/avslutar sekvenser i pågående batch (iteration-level scheduling). Orca, vLLM; undviker idle GPU vid varierande genereringslängder.
**Disaggregated Prefill-Decode** — Separata workers för compute-tung prefill vs latens-känslig decode. SplitWise; optimerar resursallokering; prefill på compute-GPU, decode på bandwidth-optimerade enheter.
**Model Parallelism for LLM** — Tensor/pipeline/expert parallellism för att träna/serva modeller > en GPU. Megatron, DeepSpeed; kombineras ofta med data parallel för trillion-parameter skala.
**Pipeline Parallelism** — Delar lager över enheter; microbatching för att hålla pipeline full. GPipe; bubble overhead vid få microbatches; 1F1B schedule minskar idle tid.
**Expert Parallelism** — Distribuerar MoE-experter över noder med all-to-all kommunikation. Varje token routas till expert på annan nod; kommunikation kan bli flaskhals.
**Sequence Parallelism** — Delar sekvensdimension över enheter vid lång kontext-träning. Ring attention, context parallel; nödvändigt när sekvens inte får plats på en GPU.
**FSDP** — Fully Sharded Data Parallel: shardar parametrar, gradienter och optimizer-states. PyTorch native; minskar per-GPU minne; all-gather vid forward/backward.
**ZeRO Optimization** — DeepSpeed Zero Redundancy Optimizer; minskar minnesduplicering i distribuerad träning. Stage 1–3 shardar optimizer, gradienter och parametrar; möjliggör träning av 100B+ modeller.
**Activation Checkpointing** — Rekomputera activations i backward för att träna större modeller på samma minne. Tradeoff compute vs minne; standard för LLM-pretraining på begränsad hårdvara.
**Loss Spike** — Plötslig ökning av träningsförlust; kan indikera instabilitet eller dålig batch. Kan kräva LR-reduktion, gradient clipping eller batch skip; vanligt i early training.
**Loss Divergence** — Förlust växer okontrollerat; ofta learning rate eller numerisk instabilitet. NaN i loss kräver omstart från checkpoint med lägre LR; mixed precision kan orsaka overflow.
**Learning Rate Warmup** — Linjärt ökar LR från nära noll; kritisk för stabila LLM-pretrain. 1–5% av totala steg; förhindrar stora initiala uppdateringar som destabiliserar träning.
**Weight Tying** — Delar vikter mellan input embedding och output LM-head; minskar parametrar. Press & Wolf; embedding och unembedding delar matris; standard i de flesta LLM:er.
**Tied Embeddings** — Samma matris för token lookup och logits-projektion. Halverar embedding-parametrar; antagande att input och output representationer lever i samma rum.
**Vocabulary Expansion** — Lägger till tokens för kod, språk eller specialtecken efter initial träning. Nya rader i embedding-matris; kräver finjustering för att lära nya token-representationer.
**Embedding Resizing** — Utökar embedding-matris vid ny vokabulär; kräver ofta finjustering. Nya rader initialiseras med medelvärde eller random; output layer måste också expanderas.
**Tokenizer Mismatch** — Fel när tränings- och inferens-tokenizer skiljer sig; ger korruptionsartefakter. Kritiskt vid modell-merge och cross-framework deployment; alltid verifiera tokenizer-kompatibilitet.
**BPE Merge Table** — Ordning av subword-sammanslagningar som definierar tokenisering. Merge-regler appliceras sekventiellt; olika merge table = olika tokenisering av samma text.
**Special Tokens** — Reserverade tokens för BOS, EOS, PAD, mask, verktyg och roller. Utanför vanlig vokabulär; adderas till embedding; måste hanteras konsekvent i träning och inferens.
**BOS Token** — Beginning-of-sequence markerar start; ibland implicit i chat templates. Vissa modeller prepend automatiskt; chat templates kan ersätta explicit BOS.
**EOS Token** — End-of-sequence signalerar att modellen ska sluta generera. Flera EOS-tokens i chat models (t.ex. per roll); stop sequence vid inferens.
**PAD Token** — Padding till fix batch-längd; måste maskeras i attention. PAD får aldrig bidra till loss eller attention; ofta samma som EOS i decoder-only modeller.

## Fine-tuning & Adaptation

**Fine-Tuning** — Fortsatt träning av förtränade vikter på uppgiftsspecifik data med lägre learning rate. Bygger på generella representationer från pretrain; typiskt sista steget före deployment för domänspecifika applikationer.
**Full Fine-Tuning** — Uppdaterar alla modellparametrar; maximalt utrymme men dyrt i minne och risk för catastrophic forgetting. Kräver multi-GPU för stora modeller; bäst när domän skiljer sig kraftigt från pretrain.
**Parameter-Efficient Fine-Tuning** — PEFT: uppdaterar liten delmängd parametrar (LoRA, adapters) med jämförbar prestanda. Möjliggör finjustering av 70B-modeller på en GPU med QLoRA; flera adapters kan kombineras.
**LoRA** — Low-Rank Adaptation: tränar lågrangsmatriser ΔW=BA som adderas till frysta vikter. Hu et al.; typiskt r=8–64; tränar <1% av parametrar med minimal kvalitetsförlust.
**LoRA Rank** — Rang r i lågranksuppdatering; högre r ger mer kapacitet men fler träningsparametrar. r=8 räcker ofta; r=64+ för komplexa uppgifter; tradeoff kapacitet vs minne.
**LoRA Alpha** — Skalingsfaktor för LoRA-uppdatering; styr effektiv learning rate för adaptern. Typiskt alpha=2r; högre alpha ger starkare adapter-effekt utan att ändra bas-LR.
**QLoRA** — Kvantiserad basmodell (4-bit) med LoRA-träning; möjliggör finjustering på konsument-GPU. Dettmers et al.; NF4-kvantisering + double quantization; 65B på en 48GB GPU.
**DoRA** — Weight-Decomposed Low-Rank Adaptation; separerar magnitud och riktning för bättre stabilitet. Liu et al.; finjusterar riktning via LoRA och magnitud separat; närmare full FT-prestanda.
**AdaLoRA** — Adaptiv rangallokering under träning via SVD-baserad parameterbudget. Zhang et al.; allokerar mer rank till viktiga lager/vikter; dynamisk komprimering av adapter.
**IA3** — Infused Adapter: lär sig elementvisa skalningsvektorer istället för full lågranksmatris. Liu et al.; ännu färre parametrar än LoRA; skalar keys, values och FFN.
**Prefix Tuning** — Tränar kontinuerliga prefix-vektorer prepended till varje lager utan att ändra basvikter. Li & Liang; virtuella tokens i key/value; extremt parameter-effektivt.
**Prompt Tuning** — Endast input-side soft prompts tränas; extremt parameter-effektivt. Lester et al.; färre parametrar än prefix tuning; fungerar bäst på stora modeller.
**P-Tuning v2** — Djupa prompt-vektorer i varje lager; starkare än ytlig prompt tuning. Liu et al.; samma idé som prefix tuning men med optimerade prompt-embeddings per lager.
**Adapter Module** — Små bottleneck-lager insatta mellan transformer-block; task-specifik utan full finetune. Houlsby et al.; down-project → act → up-project; ~1–5% extra parametrar per adapter.
**Adapter Fusion** — Kombinerar flera adapters med viktning eller routing för multi-task. Pfeiffer et al.; väljer eller blandar adapters vid inferens; modulär multi-domain deployment.
**BitFit** — Finjusterar endast bias-termer; överraskande effektivt för vissa uppgifter. Zaken et al.; <0,1% av parametrar; fungerar för enklare domain shift och klassificering.
**LayerNorm Tuning** — Uppdaterar endast normaliseringsparametrar för lätt domain shift. Minimal parameter overhead; kan räcka för stil- och domänanpassning utan full finjustering.
**Head Tuning** — Finjusterar endast task-specifikt klassificerings/LM-head ovanpå fryst backbone. Linear probing plus; endast output-lager tränas; snabb men begränsad kapacitet.
**Selective Fine-Tuning** — Uppdaterar endast utvalda lager baserat på gradient/importance-analys. Senare lager finjusteras ofta mer; tidiga lager fryses för att bevara generella features.
**Catastrophic Forgetting** — Basförmågor försämras när modellen specialiseras på smal uppgift. McCloskey & Cohen; mitigeras med EWC, replay, multi-task träning eller lägre LR.
**Elastic Weight Consolidation** — EWC: straffar ändringar av viktiga parametrar via Fisher-information. Kirkpatrick et al.; viktiga vikter för tidigare uppgifter får högre penalty vid ny träning.
**Experience Replay** — Mixar gamla träningsdata vid kontinuerlig finjustering för att bevara kunskap. Enkel men effektiv; replay buffer med tidigare uppgiftsdata förhindrar forgetting.
**Multi-Task Fine-Tuning** — Tränar på flera uppgifter samtidigt för delad representation. T5-style; blandad batch från flera dataset; bättre generalisering men risk för task interference.
**Task Arithmetic** — Kombinerar task-vektorer (skill vectors) via addition/subtraktion i viktutrymmet. Ilharco et al.; finjusteringsdelta kan adderas/subtraheras för att kombinera färdigheter.
**Model Soups** — Medelvärdar vikter från flera finetuning-körningar för robustare generalisering. Wortsman et al.; enkel viktmedelvärde över checkpoints med olika hyperparametrar.
**WiSE-FT** — Interpolation mellan förtränade och finjusterade vikter för robusthet mot distribution shift. Wortsman et al.; viktinterpolation balanserar specialisering och generell robusthet.
**Domain-Adaptive Fine-Tuning** — Finjustering på måldomän efter generell pretrain; hanterar covariate shift. Standard för domän-specifika LLM (juridik, medicin); continued pretrain + SFT.
**Instruction Fine-Tuning** — Finjustering på instruktionsformat för bättre zero/few-shot beteende. FLAN, Alpaca; förbättrar instruktionsföljning; ofta föregångare till RLHF.
**Continual Fine-Tuning** — Sekventiell finjustering på nya uppgifter utan att glömma tidigare. Kräver replay, EWC eller adapter per uppgift; praktiskt problem vid levande produktsystem.
**Reinforcement Fine-Tuning** — RL-baserad finjustering med reward signal efter SFT. PPO med reward model; förbättrar kvalitet bortom ren imitation av demonstrationsdata.
**Direct Preference Optimization** — DPO: optimerar preferenser direkt utan explicit reward model. Rafailov et al.; enklare pipeline än RLHF; implicit reward via referensmodell-KL.
**Odds Ratio Preference Optimization** — ORPO: kombinerar SFT och preferensoptimering i ett steg. Hong et al.; ingen separat SFT-fas; odds ratio-straff för rejected svar.
**Kahneman-Tversky Optimization** — KTO: preferenslärande från binära good/bad labels utan parvis jämförelse. Ethayarajh et al.; prospect theory-inspirerad loss; enklare datainsamling än pairwise.
**Supervised Fine-Tuning** — SFT: standard cross-entropy på demonstrationsdata före alignment. Första steget i RLHF-pipeline; lär modellen grundläggande svarformat och uppgiftsföljning.
**Curriculum Fine-Tuning** — Ordning av träningsdata från enkel till svår för stabilare konvergens. Enklare instruktioner först; gradvis svårare exempel; kan förbättra slutlig kvalitet.
**Data Selection for Fine-Tuning** — Urval av högkvalitativa/informativa exempel; minskar brus och kostnad. LIMA visar att få kvalitativa exempel kan räcka; kvalitet > kvantitet.
**LIMA Hypothesis** — Hypotes att få högkvalitativa exempel kan räcka för stark instruction tuning. Zhou et al.; 1000 curated exempel matchade full Alpaca; datakvalitet avgörande.
**Overfitting in Fine-Tuning** — Modellen memorerar träningsinstruktioner och generaliserar dåligt till nya formuleringar. Vanligt vid små dataset och många epochs; early stopping och regularisering kritiska.
**Early Stopping in Fine-Tuning** — Avbryter när valideringsmetrik försämras; förhindrar overfitting. Validera på held-out instruktioner med ny formulering; inte samma data som träning.
**Learning Rate for Fine-Tuning** — Typiskt 10–100× lägre än pretrain; för högt LR förstör basrepresentationer. 1e-5 till 2e-4 vanligt för full FT; högre för LoRA-only.
**Weight Decay in Fine-Tuning** — Regularisering som begränsar stora viktändringar från basmodellen. Mindre aggressiv än pretrain; viktigt för full FT; mindre relevant för LoRA.
**Gradient Accumulation in Fine-Tuning** — Simulerar större batch när GPU-minne begränsar batch size. Ackumulera över microbatches; effektiv batch = microbatch × accumulation steps.
**Mixed Precision Fine-Tuning** — BF16/FP16 träning med FP32 master weights för hastighet. Standard i modern finjustering; BF16 föredras för stabilitet; QLoRA använder 4-bit bas.
**DeepSpeed Fine-Tuning** — Distribuerad finjustering med ZeRO och offload för stora modeller. ZeRO-3 + CPU offload; möjliggör full FT av modeller som annars inte får plats.
**FSDP Fine-Tuning** — Fully sharded finjustering som minskar per-GPU minneskrav. PyTorch native; shardar parametrar och optimizer states; populärt alternativ till DeepSpeed.
**Multi-GPU Fine-Tuning** — Data parallel träning med synkroniserade gradienter över enheter. DDP eller FSDP; linear speedup med antal GPU:er; batch size skalas proportionellt.
**Single-GPU Fine-Tuning** — PEFT/QLoRA möjliggör finjustering av stora modeller på en konsument-GPU. 7B–70B med QLoRA på 24–48GB; praktiskt för prototyping och små team.
**Fine-Tuning Dataset Format** — JSONL med instruction/input/output eller messages-format enligt modell. Måste matcha chat template; inkonsistent format ger dålig konvergens.
**ChatML Format** — Strukturerat meddelandeformat med role/content för konversationsfinjustering. OpenAI-kompatibelt; <|im_start|> och  tokens; standard för många open models.
**Alpaca Format** — instruction/input/output tripplar populära i open-source finjustering. Taori et al.; enkel JSON-struktur; input optional för uppgifter utan extra kontext.
**ShareGPT Format** — Konversationsloggar som träningsdata för chat-modeller. Multi-turn dialog; kräver rensning och kvalitetsfilter; risk för låg kvalitet i scraped data.
**Packaged Fine-Tuning** — Packar korta exempel till fix sekvenslängd för effektiv träning. Sequence packing med attention mask; högre GPU-utnyttjande; undviker padding-waste.
**Truncation Strategy** — Hur långa exempel klipps; påverkar vilken information som bevaras. Trunkera input vs output; "truncate from left" för långa kontexter; förlorar ofta viktig början eller slut.
**Max Sequence Length** — Övre gräns på tokens per träningsexempel; måste matcha modellkapacitet. 2048–8192 vanligt; längre kräver lång-kontext modell och mer minne.
**Label Masking** — Maskerar förlust på prompt-delen så endast svar bidrar till gradient. Standard i instruction tuning; modellen ska inte tränas att prediktera instruktionen.
**Assistant-Only Loss** — Beräknar loss endast på assistant-svar i konversationer. User/system-meddelanden maskeras; modellen lär sig endast generera svar, inte replikera user input.
**NEFTune** — Lägger till brus i embeddings under träning; förbättrar ibland generalisering. Jain et al.; noise under finjustering som regularisering; enkel implementation, ovanlig effekt.
**Dropout During Fine-Tuning** — Regularisering; ofta lägre dropout än pretrain för att bevara kapacitet. Många sätter dropout=0 vid finjustering; modellen redan regulariserad via pretrain.
**Freezing Layers** — Fryser tidiga lager och finjusterar senare; vanligt i transfer learning. Tidiga lager fångar generella features; senare lager domänspecifika; gradvis unfreezing alternativ.
**Gradual Unfreezing** — Frigör lager stegvis uppifrån och ned för stabil domain adaptation. Howard & Ruder; en lagergrupp i taget; minskar risk för att förstöra basrepresentationer.
**Differential Learning Rates** — Olika LR per lagergrupp; lägre LR för tidiga lager. Discriminative fine-tuning; senare lager får högre LR; bevarar generella features i tidiga lager.
**Task-Specific Head** — Separat output-lager per uppgift ovanpå delad encoder. Multi-task med delad backbone; varje uppgift har eget klassificerings/LM-head.
**Multi-Adapter Routing** — Väljer adapter per uppgift eller input vid inferens. AdapterHub; dynamisk laddning; en basmodell med flera task-adapters.
**Merge Adapters into Base** — Slår ihop tränade LoRA/adapter-vikter i basmodell för enklare deployment. Eliminerar adapter-runtime overhead; W' = W + α/r·BA; permanent fusion.
**LoRA Merge** — Beräknar W' = W + α/r·BA och sparar som en modell utan adapter-runtime. Enklare deployment; ingen adapter-modul vid inferens; merged weights kan kvantiseras.
**Fine-Tuning Evaluation** — Håll separat valideringsset som inte överlappar träningsinstruktioner. Testa på nya formuleringar av samma uppgift; benchmark på held-out tasks.
**Held-Out Task Evaluation** — Testar på helt nya uppgiftsformat för att mäta generalisering. Undvik benchmark leakage; MMLU, HumanEval etc. som separata utvärderingar.
**Benchmark Leakage in Fine-Tuning** — Risk att val/test-exempel hamnar i finjusteringsdata. Inflated benchmark scores; deduplicera mot benchmark-data före träning.
**Synthetic Fine-Tuning Data** — AI-genererade exempel för att skala domänspecifik finjustering. GPT-4 genererar instruktioner; kvalitetsfilter och diversity kritiska; risk för model collapse.
**Distillation Fine-Tuning** — Student finjusteras mot teacher-logits eller svar på samma data. Mjukare träningsmål än hard labels; komprimerar teacher-kunskap till mindre modell.
**Self-Play Fine-Tuning** — Modell genererar och filtrerar egna träningsdata iterativt. STaR, ReST; modell förbättras genom egna bästa svar; kräver verifierare eller filter.
**Rejection Sampling Fine-Tuning** — Sample många svar, behåll bästa enligt verifierare, finjustera på dem. ReST; skalar finjusteringsdata med modellens egna bästa outputs.
**Iterative Fine-Tuning** — Upprepade finjusteringscykler med expanderande dataset. Varje iteration lägger till nya data; risk för drift och overfitting utan basankare.
**Fine-Tuning for Code** — Domänspecifik finjustering med kod-exempel, syntax och repo-kontext. HumanEval, MBPP; kod-specifik tokenisering; execution feedback kan förbättra kvalitet.
**Fine-Tuning for Math** — Träning på steg-för-steg-lösningar och verifierbara svar. GSM8K-style; process supervision förbättrar multi-step reasoning; verifierbar reward vid RL.
**Fine-Tuning for Tool Use** — Tränar modellen att välja verktyg och korrekt JSON-argument. Function calling dataset; strukturerad output; integration med API-schemas.
**Fine-Tuning for Safety** — Data och objektiv som minskar skadligt eller icke-efterlevande beteende. Red team data, refusal examples; kan konfliktera med helpfulness; alignment tax.
**Negative Example Fine-Tuning** — Inkluderar avsiktligt dåliga svar märkta som fel för kontrast. Unlikelihood training; modellen lär sig vad den inte ska generera; DPO rejected samples.
**Contrastive Fine-Tuning** — Optimerar representationer så positiva ligger nära och negativa långt. SimCSE-style; mindre vanligt för generativa LLM men används i embedding-modeller.
**Unlikelihood Training** — Straffar sannolikhet för oönskade sekvenser under finjustering. Welleck et al.; explicit penalty på dåliga outputs; komplement till positive SFT.
**Fine-Tuning Checkpoint Selection** — Väljer checkpoint med bäst validering, inte sista epoch. Sista epoch overfittar ofta; spara checkpoints varje epoch; evaluera på held-out val.
**Hyperparameter Search for Fine-Tuning** — Söker LR, epoch, rank, batch size på valideringsmetrik. LoRA rank och alpha; typiskt få trials räcker; grid över LR och epochs mest kritisk.
**Epoch vs Step Budget** — Träningsbudget i pass genom data vs fix antal gradientsteg. 1–3 epochs vanligt för SFT; fler epochs riskerar overfitting; step budget mer reproducerbart.
**Warmup in Fine-Tuning** — Kort LR-uppvärmning även vid finjustering för stabilitet. 3–10% av totala steg; mindre kritisk än pretrain men hjälper vid full FT.
**Cosine Decay Fine-Tuning** — LR-schema som mjuknar mot slutet av finjusteringskörning. Cosine annealing till ~0; ger finare konvergens i slutet av träning.
**Fine-Tuning Registry** — Versionering av dataset, hyperparametrar och checkpoints per experiment. MLflow, W&B; reproducerbarhet kräver spårning av data, config och modellversioner.
**Adapter Hub** — Bibliotek av delbara task-adapters som laddas ovanpå basmodell. HuggingFace PEFT hub; community adapters; en basmodell, många specialiseringar.
**Modular Fine-Tuning** — Separata moduler per färdighet som komponeras vid inferens. Mixture of adapters; routing eller ensemble; undviker multi-task interference.
**Fine-Tuning Drift** — Gradvis beteendeförändring vid upprepad finjustering utan basankare. Modellen glider från basförmågor; mitigeras med KL-straff, replay eller WiSE-FT.
**Base Model Anchoring** — Regularisering mot ursprungliga vikter för att begränsa drift. KL till referens, EWC, weight interpolation; balanserar specialisering och basförmåga.

## Alignment & RLHF

**AI Alignment** — Design av AI-system vars mål och beteende stämmer med mänskliga värderingar och avsikter. Omfattar RLHF, constitutional AI och säkerhetsforskning; central utmaning vid deployment av kapabla modeller.
**Reinforcement Learning from Human Feedback** — RLHF: tränar policy med reward model tränad på mänskliga preferenser. Christiano et al.; standard pipeline för chat-LLM: SFT → RM → PPO; förbättrar helpfulness och safety.
**Reward Model** — Modell som predicerar mänsklig preferens eller kvalitetspoäng för modelloutputs. Tränas på pairwise comparisons; blir proxy för mänsklig bedömning i RL-loop.
**Preference Dataset** — Par av svar där människor markerat vilket som är bättre; grund för alignment. Chosen vs rejected; kvalitet och diversity i preferensdata avgör RM-kvalitet.
**Bradley-Terry Model** — Probabilistisk modell för parvisa preferenser: P(a>b) = σ(r(a)-r(b)). Grund för pairwise ranking loss; antar transitiva preferenser; enkel men effektiv modell.
**Pairwise Ranking Loss** — Förlust som uppmuntrar reward model att ranka preferred svar högre. -log σ(r(chosen) - r(rejected)); standard RM-träningsmål; kan utökas med margin.
**Proximal Policy Optimization** — PPO: policy gradient-algoritm med klippad objektivfunktion för stabil RL. Schulman et al.; standard i RLHF; klipper policy ratio för att begränsa stora uppdateringar.
**RLHF Pipeline** — Typiskt: SFT → reward model → PPO fine-tuning med KL-straff mot referenspolicy. Tre steg; varje steg kan itereras; DPO förenklar till två steg utan explicit RL.
**KL Penalty to Reference** — Straffar avvikelse från SFT-policy för att undvika reward hacking och mode collapse. KL till fryst SFT-modell; β styr styrka; för svagt β → reward hacking.
**Reward Hacking** — Agent utnyttjar brister i reward model för högt score utan önskat beteende. RM favoriserar längd, formatering eller specifika fraser; KL-straff och RM-ensemble mitigerar.
**Goodhart's Law in RLHF** — När reward blir mål förlorar den sin validitet som mått på verklig kvalitet. Optimera RM-score ≠ optimera mänsklig preferens; RM overoptimization är central risk.
**Constitutional AI** — Modell kritiserar och reviderar egna svar enligt definierade principer. Anthropic; självförbättring utan mänsklig feedback per exempel; RLAIF med AI-judge.
**RLAIF** — Reinforcement Learning from AI Feedback: AI istället för människor som preferensjudge. Skalbar alternativ till RLHF; AI-judge kan ha egna bias; används i Constitutional AI.
**Direct Preference Optimization** — DPO: direkt preferensoptimering utan explicit RL-loop. Rafailov et al.; closed-form loss från preferenser; enklare och mer stabil än PPO i praktiken.
**Identity Preference Optimization** — IPO: variant av DPO med stabilare objektiv vid brusiga preferenser. Azar et al.; reglerar sigmoid för att undvika overconfidence; bättre vid noisy labels.
**SimPO** — Simple Preference Optimization: reference-free preferensmetod med sequence-level reward. Meng et al.; ingen referensmodell; length-normalized reward; enklare pipeline.
**Kahneman-Tversky Optimization** — KTO: optimerar från binära labels utan parvis jämförelse. Ethayarajh et al.; prospect theory-inspirerad; good/bad labels istället för pairwise ranking.
**ORPO** — Odds Ratio Preference Optimization: kombinerar SFT och preferenser i ett träningssteg. Hong et al.; odds ratio-straff eliminerar separat SFT-fas; effektivare pipeline.
**Rejection Sampling** — Generera många svar, filtrera med reward model, finjustera på bästa. ReST; skalar alignment-data; modellens egna bästa outputs blir träningsdata.
**Best-of-N Policy** — Väljer bästa av N samples enligt reward; ökar kvalitet utan viktuppdatering. Inference-time alignment; kostnad skalar med N; alternativ till RL fine-tuning.
**Online RLHF** — Iterativ loop: samla data från aktuell policy → uppdatera reward → träna policy. On-policy data; RM tränas på aktuell policy outputs; dyrare men mer relevant data.
**Offline RLHF** — Tränar på statisk preferensdata utan interaktiv datainsamling. Enklare pipeline; distribution shift mellan SFT och policy kan degradera RM; DPO är offline-metod.
**Human Preference Elicitation** — Metoder för att samla in jämförelser, rankningar eller skala-betyg. Pairwise comparison mest informativt per annotering; Likert-skala enklare men mindre discriminativt.
**Annotator Disagreement** — Variation mellan mänskliga bedömningar; kräver aggregation och kvalitetskontroll. Majority vote, MACE eller viktning efter annotatorkvalitet; disagreement signalerar ambiguous prompts.
**Inter-Rater Reliability** — Mått på konsistens mellan annotatörer (Cohen's kappa m.fl.). Låg IRR indikerar otydliga instruktioner eller subjektiva preferenser; krävs för tillförlitlig RM-träning.
**Red Teaming** — Systematisk adversarial testning för att hitta skadliga eller oönskade svar. Expert-prompts designade för att kringgå säkerhet; data för safety fine-tuning och utvärdering.
**Adversarial Prompting** — Prompts designade för att kringgå säkerhetsbegränsninger. Jailbreak-försök; DAN, roleplay, encoding tricks; kontinuerlig katt-och-råtta med alignment.
**Jailbreak** — Framgångsrik bypass av modellens säkerhetsinstruktioner. Modellen genererar skadligt innehåll trots alignment; nya jailbreaks upptäcks kontinuerligt post-deployment.
**Refusal Behavior** — Modellens tränade avvisning av skadliga eller otillåtna förfrågningar. "I can't help with that"; tränas via refusal examples och harmlessness reward; balans mot over-refusal.
**Over-Refusal** — Modellen avvisar legitima förfrågningar p.g.a. överkonservativ alignment. Falska positiva på säkerhetsfilter; alignment tax; användarfrustration vid för strikta modeller.
**Helpfulness vs Harmlessness** — Tradeoff mellan att vara användbar och att undvika skada. Anthropic HH-RLHF; multi-objective optimization; förbättra en dimension kan försämra den andra.
**Helpful Assistant Objective** — Optimerar för användarnytta inom säkerhetsgränser. Följ instruktioner, ge korrekta svar, var engagerad; primärt mål för chat-LLM utöver safety.
**Harmlessness Training** — Data och reward som straffar skadligt, olagligt eller vilseledande innehåll. Refusal training, red team data; explicit säkerhetsdimension i multi-objective RLHF.
**Honesty Alignment** — Strävar efter sanningsenlighet och erkännande av osäkerhet. Undviker fabrication; erkänner okunskap; TruthfulQA som benchmark; konflikter ibland med helpfulness.
**Sycophancy** — Tendens att hålla med användaren även när det är felaktigt. RLHF kan förstärka sycophancy via preferensdata; användare belönar bekräftande svar; aktivt forskningsproblem.
**Value Alignment** — Modellens beteende överensstämmer med specificerade värderingar och normer. Kulturellt beroende; vems värderingar ska optimeras; normativ osäkerhet central utmaning.
**Normative Uncertainty** — Osäkerhet om vilka värderingar som ska optimeras i pluralistiska samhällen. Ingen konsensus om "rätt" beteende; alignment måste hantera värdekonflikter och kulturell variation.
**Moral Uncertainty in AI** — Hur agenter ska agera när etiska principer konflikter. Trolley-problem skalat; maskininlärning kan inte lösa filosofiska frågor men måste hantera tradeoffs.
**Scalable Oversight** — Metoder för att övervaka superhuman AI med begränsad mänsklig insats. Debate, recursive reward modeling; människor kan inte direkt bedöma superhuman outputs.
**Debate** — Två AI-agenter debatterar; människa bedömer vinnare för att extrahera sanning. Irving et al.; sanning som Nash equilibrium i debate; teoretiskt ramverk för scalable oversight.
**Recursive Reward Modeling** — Människor bedömer AI som hjälper bedöma mer komplexa outputs. Leike et al.; hierarkisk bedömning; AI sammanfattar för mänsklig utvärdering.
**Iterated Amplification** — Upprepad distillation av mänsklig+AI-bedömning till mer kapabel agent. Christiano et al.; ALIGNED AGENT via iterative distillation; teoretisk path till superhuman alignment.
**Process Supervision** — Belönar korrekta mellansteg i resonemang, inte bara slutgiltigt svar. Lightman et al.; PRM betygsätter varje steg; bättre för matematik och logik än outcome-only.
**Outcome Supervision** — Belönar endast korrekt slutresultat; enklare men svagare signal. Modellen kan nå rätt svar med felaktigt resonemang; outcome RM enklare att träna men mindre informativ.
**Process Reward Model** — PRM betygsätter varje steg i kedja av tanke. OpenAI PRM800k; step-level labels; används i reasoning models som o1; dyrare att annotera.
**Outcome Reward Model** — ORM betygsätter komplett svar. Enklare annotation; standard i klassisk RLHF; kan belöna lucky guesses utan korrekt reasoning chain.
**Verifiable Reward** — Automatiskt checkbar reward (t.ex. enhetstester för kod, matematisk verifiering). Objektiv reward utan mänsklig bedömning; RLVR för kod och matematik; skalbar och objektiv.
**RL with Verifiable Rewards** — RLVR: RL där reward kommer från objektiv verifiering. DeepSeek-R1, OpenAI o1; unit tests, symbolic math verification; undviker RM-bias.
**Constitutional Principles** — Skriftliga regler som styr självrevision och träningsdata. Anthropic constitution; modell utvärderar svar mot principer; skalbar alignment utan per-exempel mänsklig feedback.
**Critique-Revision Loop** — Modell genererar kritik och förbättrat svar iterativt. Constitutional AI; self-critique → revision → final output; träningsdata genereras via denna loop.
**Self-Critique** — Modellen utvärderar egna svar mot kriterier innan final output. Constitutional AI; modell identifierar problem i eget svar; förbättrar utan extern feedback.
**Safety Fine-Tuning** — Finjustering specifikt på säkerhets- och policy-exempel. Refusal examples, red team data; separat från helpfulness tuning; kan orsaka over-refusal.
**Toxicity Reduction** — Träning/reward som minskar hat, trakasserier och grovt språk. Perspective API som filter; toxicity classifier i reward; viktigt för deployment i produkt.
**Bias Mitigation in Alignment** — Åtgärder för att minska stereotyper och unfair behandling. Balanced training data, debiasing prompts, fairness metrics; inget perfekt; pågående forskning.
**Preference Model Calibration** — RM:s scores ska korrelera med faktisk mänsklig preferens. Kalibrerade sannolikheter; RM overconfidence leder till reward hacking; Platt scaling på RM outputs.
**Reward Model Overoptimization** — Policy utnyttjar RM:s blind spots när KL-straff är för svagt. Gao et al.; test loss minskar medan human eval försämras; kritisk RLHF-fälla.
**Gold Standard Human Eval** — Högkvalitativa mänskliga bedömningar som referens för automatiska metriker. Dyrt men nödvändigt; LLM-as-judge korrelerar men ersätter inte människor helt.
**LLM-as-Judge** — Använder LLM för att bedöma svar; skalbart men med egna bias. GPT-4 som judge; position bias, length bias; billigare än mänsklig eval men kräver kalibrering.
**Judge Prompt Design** — Instruktioner till judge-LLM som påverkar bedömningens validitet. Tydliga kriterier, position randomization; dålig judge prompt ger opålitliga preferenslabels.
**Position Bias in LLM Judges** — Judge favoriserar svar i viss position i parvis jämförelse. Första svaret favoriseras; mitigera med position swap och medelvärde; systematisk bias i RLAIF.
**Length Bias in Reward** — RM eller judge favoriserar längre svar oavsett kvalitet. Längre svar får högre reward; length normalization i SimPO; explicit penalty på överdriven längd.
**Style Bias in Preferences** — Preferenser driven av ton/stil snarare än korrekthet. Formellt vs informellt språk; preferensdata reflekterar annotatorstil; svårt att separera form och innehåll.
**Multi-Objective Alignment** — Balanserar flera mål (helpful, harmless, honest) samtidigt. Pareto-optimal policies; viktning eller constraint optimization; ingen universal lösning.
**Pareto Frontier of Alignment** — Mängd av policyer där ingen måldimension kan förbättras utan att försämra annan. Visualiserar helpfulness-harmlessness tradeoff; olika punkter för olika use cases.
**Constraint Optimization in RLHF** — Maximera reward under hårda säkerhetsconstraints. Lagrangian methods; safety constraint som hard filter; Safe RLHF integrerar safety model.
**Lagrangian RLHF** — Dynamiska multiplikatorer balanserar reward vs KL/safety-straff. λ uppdateras adaptivt; balanserar competing objectives under träning.
**Safe RLHF** — Integrerar explicit säkerhetsmodell i RLHF-optimering. Dai et al.; separata helpful och safe reward; constrained PPO; minskar skadlig output vid RL.
**Toxicity Classifier Guardrail** — Filter som blockerar eller omskriver toxisk output post-hoc. Perspective API, custom classifier; sista försvarslinje efter modellgenerering; kan ge false positives.
**Moderation API** — Extern klassificerare för policyöverträdelser i input/output. OpenAI Moderation; blockera före/efter modell; komplement till träningsbaserad alignment.
**Content Policy** — Regler för tillåtet/innehåll som styr träning och deployment. Plattformsspecifik; uppdateras kontinuerligt; styr vad modellen ska vägra generera.
**Alignment Tax** — Prestationsförlust på vissa uppgifter efter säkerhets-/preferensoptimering. MMLU kan minska efter RLHF; tradeoff mellan capability och safety; debatterad storlek.
**Capability-Alignment Tradeoff** — Starkare capabilities kan öka risk om alignment inte skalar. Mer kapabel modell kan hitta fler jailbreaks; alignment måste skala med capability.
**Deceptive Alignment** — Teoretisk risk: modell appearar aligned under träning men inte i deployment. Hubinger et al.; modell lär sig att verka aligned i eval; svår att detektera.
**Inner Alignment** — Agentens interna mål matchar avsedda träningsmål. Modellen "vill" faktiskt det vi tränade den på; skiljer sig från outer alignment och deceptive alignment.
**Outer Alignment** — Specificerade träningsmål matchar designerens intentioner. Reward function fångar avsedd intent; Goodhart's law när proxy divergerar från intent.
**Mesaa Optimization** — Optimering av proxy-mått som divergerar från avsedda mål. Typo för Mesa optimization; inner misalignment; proxy blir mål (Goodhart).
**Specification Gaming** — Agent uppfyller bokstavlig specifikation men inte avsedd intent. Reward hacking specialfall; klassiskt RL-problem; RLHF är inte immunt.
**Inverse Reinforcement Learning** — IRL: infererar reward function från demonstrations. Ng & Russell; observera beteende, inferera underliggande reward; teoretisk grund för preference learning.
**Cooperative Inverse RL** — Människa och agent samarbetar; agent infererar mänskliga mål. Hadfield-Menell et al.; agent osäker på mänskliga preferenser; aktiv kommunikation.
**Human-in-the-Loop RL** — Människa ger feedback under pågående policy learning. Interaktiv preferensinsamling; dyrare men mer relevant än offline data.
**Active Preference Learning** — Väljer de jämförelser som ger mest information om preferenser. Uncertainty sampling; minimerar antal mänskliga jämförelser för given RM-kvalitet.
**Distributional Shift in RLHF** — Policy-genererad data skiljer sig från SFT-data; RM kan extrapolera dåligt. RM tränad på SFT outputs; policy genererar OOD samples; online RLHF adresserar detta.
**On-Policy vs Off-Policy RLHF** — On-policy: tränar på data från aktuell policy; off-policy: statisk data. PPO är on-policy; DPO är off-policy; tradeoff stabilitet vs datamängd.
**Reference Model in DPO** — Fryst SFT-modell som implicit KL-ankare i DPO-objektivet. Förhindrar policy från att avvika för långt från SFT; β styr styrka av ankarband.
**Beta in DPO** — Temperaturparameter som styr styrka av preferensoptimering vs KL till referens. Högre β = starkare preferensoptimering; lägre β = närmare referensmodell.
**Alignment Evaluation Suite** — Samling av säkerhets-, sanning- och hjälpsamhetstester. HarmBench, TruthfulQA, MT-Bench; ingen enskild metrik fångar alignment; multi-dimensional eval.
**Harm Benchmark** — Standardiserade prompts för att mäta skadlig modelloutput. HarmBench; kategoriserade skadliga prompts; mäter refusal rate och attack success rate.
**TruthfulQA Alignment** — Benchmark för sanningsenlighet vs populära missuppfattninger. Modellen ska undvika vanliga myter; mäter honesty dimension av alignment.
**MT-Bench Human Alignment** — Multi-turn konversationskvalitet bedömd av människor eller judge. Zheng et al.; 80 multi-turn frågor; korrelerar med human preference; standard LLM eval.
**Reward Model Ensemble** — Flera RM kombineras för robustare reward signal. Minskar enskild RM:s blind spots; medelvärde eller min score; dyrare men mer robust RLHF.

## RAG & Retrieval

**Retrieval-Augmented Generation** — RAG: hämtar externa dokument och conditionar generering på dem för aktuell kunskap. Lewis et al.; kompenserar knowledge cutoff; minskar hallucination genom evidensbaserad generering.
**Vector Database** — Databas optimerad för likhetssökning i högdimensionella embeddings. Pinecone, Weaviate, Milvus; ANN-index; skiljer sig från traditionella DB via vektorlikhet som primär query.
**Embedding Model** — Modell som mappar text till dense vektorer för semantisk likhetssökning. E5, BGE, OpenAI embeddings; kvalitet på retrieval avgörs av embedding-modellens träning och domänmatchning.
**Dense Retrieval** — Hämtning via embedding-likhet istället för lexikal keyword-matchning. Fångar semantisk likhet; missar exakta keyword-matchningar; kompletteras ofta med sparse retrieval.
**Sparse Retrieval** — Klassisk IR med termvikter (BM25, TF-IDF) utan neural embeddings. Exakt keyword-matchning; missar semantisk parfrasering; fortfarande stark baseline och komplement till dense.
**Hybrid Retrieval** — Kombinerar sparse och dense scores för robustare recall. Reciprocal Rank Fusion eller viktad kombination; fångar både lexikal och semantisk matchning.
**BM25** — Probabilistisk rankingfunktion baserad på termfrekvens och dokumentlängdsnormalisering. Robertson et al.; de facto sparse retrieval baseline; robust och tolkbar utan träning.
**TF-IDF** — Term frequency-inverse document frequency; enkel viktning av viktiga termer. Enklare än BM25; fortfarande användbar baseline; saknar BM25:s dokumentlängdsnormalisering och saturationsfunktion.
**Approximate Nearest Neighbor Search** — ANN: snabb ungefärlig sökning i stora vektorindex (HNSW, IVF). Tradeoff recall vs hastighet; exakt sökning O(n) oacceptabelt för miljontals vektorer.
**HNSW Index** — Hierarchical Navigable Small World: graf-baserat ANN-index med hög recall. Malkov & Yashunin; låg latens och hög recall; standardval för de flesta vektordatabaser.
**IVF Index** — Inverted File Index: klustrar vektorer och söker i relevanta kluster. Coarse quantizer + fine search; snabbare men lägre recall än HNSW; bra för mycket stora index.
**Product Quantization** — PQ: komprimerar vektorer för minnes- och sök-effektivitet med liten precisionförlust. Jégou et al.; delar vektor i subvektorer och kvantiserar separat; möjliggör billion-scale index.
**Cosine Similarity** — Mått på vinkel mellan vektorer; standard för normaliserade embeddings. Ovanligt för onormaliserade vektorer; ekvivalent med dot product vid unit vectors.
**Dot Product Retrieval** — Inner product som likhet; kräver ofta normaliserade embeddings. Snabbare att beräkna än cosine om vektorer redan normaliserade; MIPS (Maximum Inner Product Search).
**Euclidean Distance in Retrieval** — L2-avstånd som alternativ likhetsmetrik i vektorrum. Ekivalent med cosine för normaliserade vektorer; vissa embedding-modeller tränade med L2.
**Chunking Strategy** — Delar dokument i segment för indexering; påverkar recall och precision. För stora chunks → imprecise retrieval; för små → förlorad kontext; domänberoende optimal storlek.
**Fixed-Size Chunking** — Delar text i fix token-/teckenlängd med eventuell overlap. Enkelt och förutsägbart; kan klippa meningar och stycken mitt; 256–512 tokens vanligt.
**Semantic Chunking** — Delar vid naturliga avsnittsgränser baserat på struktur eller embeddings. Respekterar dokumentstruktur; bättre chunk-sammanhang; dyrare att implementera.
**Parent-Document Retriever** — Hämtar liten chunk men returnerar större omgivande kontext. Small-to-big retrieval; precis sökning på chunk, rik kontext vid generering.
**Sliding Window Chunking** — Överlappande fönster för att undvika att information klipps vid gränser. 10–20% overlap vanligt; duplicerar gränsinformation; ökar indexstorlek marginellt.
**Document Ingestion Pipeline** — ETL: ladda, rensa, chunka, embedda och indexera källor. PDF-parsing, HTML-rensning, metadata-extraktion; kvalitet in → kvalitet ut i RAG.
**Metadata Filtering** — Filtrerar retrieval på metadata (datum, källa, taggar) före likhetssökning. Pre-filter eller post-filter; begränsar sökrymden; viktigt för multi-tenant och tidskänsliga frågor.
**Multi-Vector Retrieval** — Flera embeddings per dokument (ColBERT-style) för finare matchning. Token- eller passage-nivå vektorer; högre recall men större index och dyrare sökning.
**ColBERT** — Late interaction: token-nivå embeddings med MaxSim-scoring vid retrieval. Khattab & Zaharia; finare matchning än single-vector; dyrare men state-of-art på BEIR.
**Cross-Encoder Reranker** — Joint encoding av query+doc för exakt relevance score; dyr men precis. BERT-baserad; O(n) forward passes för n kandidater; andra steg efter bi-encoder retrieval.
**Bi-Encoder Retrieval** — Separata encoders för query och doc; snabb ANN-sökning. DUAL-encoder; pre-computed doc embeddings; snabb men mindre exakt än cross-encoder.
**Reranking Stage** — Andra pass som sorterar top-k kandidater med kraftfullare modell. Retrieve 100, rerank to 5; cross-encoder eller LLM reranker; dramatisk precision-förbättring.
**Retrieve-then-Read** — Klassisk RAG: hämta dokument → mata in i LLM som kontext. Lewis et al.; enkel pipeline; LLM läser och genererar svar baserat på hämtad kontext.
**Retrieve-then-Generate** — Generering conditionad på hämtade passager utan explicit reader-modell. End-to-end med LLM som både reader och generator; standard modern RAG.
**Query Rewriting** — Omformulerar användarfråga för bättre retrieval (HyDE, step-back). LLM expanderar eller förenklar query; förbättrar recall vid vag eller komplex fråga.
**HyDE** — Hypothetical Document Embeddings: genererar hypotetiskt svar och söker med dess embedding. Gao et al.; query-doc asymmetri minskas; hypotetiskt svar närmare relevant doc i embedding-space.
**Multi-Query Retrieval** — Genererar flera query-varianter och union av resultat. RAG-Fusion; paraphrases och sub-queries; ökar recall; deduplicering av resultat krävs.
**Step-Back Prompting** — Genererar mer abstrakt fråga parallellt för bredare retrieval. Zheng et al.; "step back" till principfråga; kompletterar specifik fråga med generell kontext.
**Self-RAG** — Modellen beslutar när den ska hämta, relevansbedöma och kritisera egna svar. Asai et al.; reflection tokens; adaptiv retrieval; modellen kontrollerar RAG-loopen.
**Corrective RAG** — CRAG: verifierar retrieval-kvalitet och kan söka om eller använda web fallback. Yan et al.; retrieval evaluator; web search som fallback vid låg konfidens.
**Adaptive RAG** — Väljer retrieval-strategi baserat på frågetyp och konfidens. Ingen retrieval för enkla frågor; multi-hop för komplexa; sparar latens och minskar brus.
**GraphRAG** — Bygger kunskapsgraf från korpus för strukturerad community-baserad retrieval. Microsoft; community summaries; bättre för holistiska frågor över hela korpus.
**Agentic RAG** — Agent planerar flera retrieval-steg och verktygsanrop dynamiskt. ReAct-style; iterativ retrieval; komplexa frågor kräver flera sökningar och syntes.
**Multi-Hop Retrieval** — Flera sekventiella sökningar för att samla bevis över dokument. HotpotQA-style; första retrieval informerar andra query; agentic eller pipeline-baserat.
**Fusion-in-Decoder** — FiD: encoder processar flera docs separat; decoder fuserar dem. Izacard & Grave; bättre än concat i prompt; varje doc separat encoderad.
**Context Window Budget** — Max tokens för hämtade docs + prompt; kräver prioritering/truncation. Top-k docs måste passa i kontext; reranking prioriterar; compression som alternativ.
**Lost in the Middle** — LLM ignorerar information i mitten av lång kontext; påverkar RAG-layout. Liu et al.; placera viktigaste docs först och sist; U-shaped attention pattern.
**Context Ordering** — Placering av viktigaste docs först/sist kan förbättra att de används. Motverkar lost-in-the-middle; reranker-sortering bör ta hänsyn till position bias.
**Citation in RAG** — Modellen refererar källor; kräver träning eller post-processing. Inline citations [1], [2]; verifierbar attribution; förbättrar trust och verifierbarhet.
**Attribution Evaluation** — Mäter om genererade påståenden stöds av hämtade källor. AIS, QAEval; automatisk eller manuell; kritiskt för tillförlitlig RAG.
**Faithfulness in RAG** — Output följer hämtad evidens utan fabricerade tillägg. RAGAS faithfulness metric; modellen ska inte hallucinera bortom kontext; kvarstående utmaning.
**Answerability Detection** — Bedömer om hämtad kontext räcker för att besvara frågan. Abstention om otillräcklig evidens; förhindrar fabricerade svar vid dålig retrieval.
**Abstention in RAG** — Modellen avstår svara när evidens är otillräcklig. "I don't have enough information"; tränas explicit; bättre än hallucination vid misslyckad retrieval.
**Hallucination in RAG** — Modellen fabricerar trots tillgänglig kontext; kvarstående risk. RAG minskar men eliminerar inte hallucination; modellen kan ignorera eller misstolka kontext.
**Grounding** — Binda svar till specifik källtext eller fakta. Evidensbaserade svar; citation och attribution; motsats till fri generering från parametrisk kunskap.
**Knowledge Cutoff** — Datum efter vilket basmodell saknar kunskap; RAG kompenserar. Primär motivation för RAG; extern kunskapsbas ger aktuell information utan omträning.
**Freshness in Retrieval** — Prioriterar nyare dokument vid tidskänsliga frågor. Metadata-baserad recency boost; viktigt för nyheter, priser och policy-dokument.
**Incremental Index Update** — Lägger till/uppdaterar vektorer utan full reindex. Nya dokument embeddas och insertas; gamla versioner markeras obsolete; undviker dyra full rebuilds.
**Embedding Drift** — Förändrad embedding-modell gör gamla index inkompatibla. Ny modell kräver full reindex; versionera embedding-modell i pipeline; migration plan vid modellbyte.
**Reindexing** — Full ombyggnad av vektorindex efter modell- eller schemaändring. Dyr operation för stora korpus; schemalagd eller triggered vid embedding-modellbyte.
**Deduplication in Corpus** — Tar bort duplicerade/near-duplicate chunks före indexering. MinHash, SimHash; minskar indexstorlek och förhindrar redundant retrieval.
**PII Redaction in Ingestion** — Maskerar persondata innan indexering och retrieval. GDPR-compliance; regex och NER-baserad redaction; innan embedding och lagring.
**Access Control in RAG** — Filtrerar retrieval per användares behörigheter. Metadata ACL; användare ser endast docs de har access till; kritiskt för enterprise RAG.
**Multi-Tenant RAG** — Isolerade index per kund med delad infrastruktur. Namespace eller separata index; data isolation; delad embedding och reranking infrastruktur.
**Retrieval Recall@k** — Andel queries där relevant doc finns i top-k. Standard retrieval-metrik; k=5 eller k=10; måste definiera "relevant" per query.
**Retrieval MRR** — Mean Reciprocal Rank: belönar högt rankad första relevanta träff. 1/rank av första relevanta; känslig för ranking-position; standard på BEIR.
**Retrieval NDCG** — Normaliserad discounted cumulative gain; rankningskvalitet med graded relevance. Graded relevance (0–3); position-discounted; guldstandard för rankningsevaluering.
**Hit Rate** — Andel queries med minst en relevant träff i top-k. Binär variant av recall@k; enklare tolkning; användbar för produktmonitoring.
**RAGAS Framework** — Automatiska metriker: faithfulness, answer relevance, context precision/recall. Es et al.; LLM-baserad eval; ingen ground truth krävs; populärt för RAG-utvärdering.
**Context Precision** — Andel hämtade chunks som faktiskt är relevanta för frågan. Hög precision → mindre brus i LLM-kontext; reranking förbättrar precision.
**Context Recall** — Andel nödvändig information som hämtades från källan. Låg recall → modellen saknar evidens; chunking och retrieval-strategi avgör.
**Answer Relevance** — Genererat svar adresserar frågan utan irrelevant innehåll. RAGAS metric; LLM bedömer om svar är relevant för query; oberoende av faithfulness.
**Noise Robustness in RAG** — Prestanda när retrieval inkluderar irrelevanta dokument. Modellen ska filtrera brus; lost-in-the-middle förvärras; reranking och instruktioner hjälper.
**Negative Retrieval** — Medvetet irrelevanta docs i kontext; testar modellens filtrering. Stress test; mäter om modellen kan ignorera distraktorer; robusthet benchmark.
**Long-Context RAG** — Hämtar många eller långa docs när LLM har stor kontext. 100k+ kontext möjliggör mer retrieval; lost-in-the-middle kvarstår; reranking ännu viktigare.
**Small-to-Big Retrieval** — Hämta små chunks, expandera till större kontext vid generering. Parent-document retriever; precis sökning, rik generering; bästa av båda världar.
**Sentence Window Retrieval** — Hämtar mening med omgivande fönster som kontext. LlamaIndex; sök på mening, generera med ±5 meningar; balanserar precision och kontext.
**Auto-Merging Retriever** — Slår ihop relaterade chunks hierarkiskt efter retrieval. LlamaIndex; parent nodes slås ihop om flera barn chunks retrievar; dynamisk kontextexpansion.
**Knowledge Graph Retrieval** — Söker i KG noder/kanter och serialiserar till LLM-kontext. GraphRAG, entity linking; strukturerad kunskap komplement till vektor-sökning.
**Structured Data RAG** — SQL/API över tabeller kombinerat med vektor-sökning. Text-to-SQL; hybrid över strukturerad och ostrukturerad data; enterprise data sources.
**Tool-Augmented RAG** — Agent anropar search/API/SQL som retrieval-verktyg. LLM väljer verktyg dynamiskt; mer flexibelt än statisk retrieve-then-read pipeline.
**Web Search RAG** — Live webbsökning som dynamisk kunskapskälla. Bing/Google API; aktuell information; latency och kvalitetskontroll utmaningar; CRAG web fallback.
**Cache-Augmented Generation** — CAG: förcomputad kunskap i KV-cache istället för runtime retrieval. Precompute KV-cache för statisk kunskapsbas; snabbare än RAG för statisk data.
**Prompt Compression for RAG** — Komprimerar hämtad kontext (LLMLingua) för att passa budget. Jiang et al.; behåller viktig information; 4× kompression med minimal kvalitetsförlust.
**Contextual Chunk Headers** — Prepender metadata/rubrik till chunk för bättre disambiguation. "Document: X, Section: Y\n{chunk}"; hjälper embedding och LLM disambiguera.
**Late Chunking** — Embedda hel dokument först, chunka i embedding-space. Jina AI; bättre kontextuella embeddings; chunks behåller dokumentkontext i representationen.
**Matryoshka Embeddings** — Truncerbara embeddings för flexibel precision/latens-tradeoff. Kusupati et al.; korta vektorer för snabb sökning, fulla för precision; en modell, flera storlekar.
**Embedding Fine-Tuning for Domain** — Domänspecifik finjustering av retriever för bättre recall. Contrastive fine-tuning på domän-QA par; avgörande för specialized RAG (juridik, medicin).
**Hard Negative Mining** — Tränar retriever med svåra felaktiga docs som negativa exempel. ANN-sökning hittar hard negatives; förbättrar discriminering; kritisk för contrastive training.
**Contrastive Retrieval Training** — InfoNCE-lik förlust som drar query nära relevant doc. In-batch negatives; triplet loss; standard för embedding-modell träning (E5, BGE).
**In-Batch Negatives** — Andra docs i batch används som negativa i contrastive träning. Gratis negatives utan extra sampling; batch size påverkar negative kvalitet.
**Query-Document Asymmetry** — Query och doc kan ha olika encoders eller instruktioner. "query: " och "passage: " prefixes; asymmetrisk arkitektur förbättrar retrieval.
**Instruction-Tuned Retriever** — Retriever tränad med instruktioner per uppgiftstyp. E5, BGE; "Represent this sentence for retrieval"; task-specific instructions vid inferens.
**Multi-Lingual Retrieval** — Cross-lingual sökning: fråga på ett språk, docs på annat. mE5, multilingual BGE; viktigt för globala RAG-system; embedding-space alignment.
**RAG Latency Budget** — Total tid för embed + search + rerank + generate. Typiskt 200ms–2s; embed ~10ms, ANN ~20ms, rerank ~100ms, generate ~1s; optimering per komponent.

## AI-agenter

**AI Agent** — Autonomt LLM-baserat system som iterativt planerar, väljer och anropar externa verktyg, tolkar deras resultat och agerar mot definierade mål över flera steg utan kontinuerlig mänsklig styrning i varje delmoment.
**Agent Loop** — Observe → plan → act → observe-cykel som upprepas tills uppgiften är klar, ett stopptillstånd nås eller fördefinierad tids-, token- eller iterationsbudget är förbrukad.
**ReAct Pattern** — Reasoning + Acting: modellen alternerar explicit resonemang i naturligt språk med strukturerade verktygsanrop i samma trace, så att varje action motiveras av synlig tankegång.
**Plan-and-Execute** — Separerar planering från exekvering; en planner genererar en ordnad steglista som en executor följer, vilket minskar omplanering och stabiliserar långa flerstegsuppgifter.
**Tool Calling** — Strukturerat API där modellen väljer verktyg, fyller i JSON-argument enligt schema och tar emot serialiserade resultat som matas tillbaka in i kontexten.
**Function Schema** — JSON Schema som beskriver verktygets namn, parametrar, typer och obligatoriska fält så att LLM kan generera validerbara anrop som runtime kan parsa deterministiskt.
**Tool Registry** — Katalog över tillgängliga verktyg med metadata, versioner, rate limits och access policies som styr vilka capabilities varje agent eller roll får använda.
**Agent Memory** — Kort- och långtidsminne för att bevara konversation, fakta, mellanresultat och tidigare erfarenheter så att agenten kan agera sammanhängande över sessioner och komplexa uppgifter.
**Short-Term Agent Memory** — Aktuell konversation, verktygsresultat och scratchpad inom modellens kontextfönster; försvinner eller komprimeras när tokenbudgeten överskrids.
**Long-Term Agent Memory** — Persistent lagring i vektor-DB, profil eller dokumentarkiv utöver en session; möjliggör återkallning av användarpreferenser och tidigare slutsatser vid nya körningar.
**Episodic Memory** — Lagrar specifika händelser och interaktioner som tidsstämplade episoder som agenten kan referera till vid liknande framtida uppgifter eller för att undvika upprepade misstag.
**Semantic Memory** — Generaliserade fakta och kunskap extraherade från erfarenheter, ofta indexerade semantiskt, till skillnad från råa episodiska loggar av enskilda händelser.
**Working Memory Scratchpad** — Explicit yta där agenten skriver mellanresultat, delplaner och hypoteser som hålls tillgängliga mellan verktygsanrop utan att blanda dem med slutligt användarsvar.
**Multi-Agent System** — Flera specialiserade agenter samarbetar via meddelanden, delad state eller orchestrator; varje agent kan ha egna verktyg, prompts och ansvarsområden.
**Orchestrator Agent** — Koordinerar sub-agenter, delegerar deluppgifter, samlar in partiella resultat och aggregerar dem till ett slutligt svar eller en gemensam action.
**Hierarchical Agents** — Manager-agenter delegerar till worker-agenter i trädstruktur; högre nivåer planerar strategi medan lägre nivåer utför atomära verktygsanrop och lokala beslut.
**Agent Communication Protocol** — Standardiserat meddelandeformat mellan agenter, t.ex. MCP eller A2A, med definierade roller, metadata och felhantering för interoperabilitet mellan leverantörer.
**Model Context Protocol** — MCP: öppet protokoll för att koppla LLM till verktyg, datakällor och externa tjänster via standardiserade resurs- och promptdefinitioner.
**Agent-to-Agent Protocol** — A2A: interoperabilitet mellan agenter från olika leverantörer via gemensamma meddelandeformat, capability discovery och säker delegering av deluppgifter.
**Handoff Between Agents** — Överlämnande av konversation, uppgift och relevant kontext från en agent till en annan specialist utan att användaren behöver upprepa hela historiken.
**Subagent Delegation** — Parent-agent spawnar child-agent med begränsat scope, verktyg och tokenbudget för att isolera risk och parallellisera deluppgifter i stora arbetsflöden.
**Agent State Machine** — Explicita tillstånd som planning, executing, waiting och error styr agentflöde, timeouts och vilka transitions som är tillåtna vid varje fas.
**Goal Decomposition** — Bryter högnivåmål i delmål och atomära actions som kan mappas till konkreta verktyg, med beroenden och ordning mellan stegen.
**Task Planning** — Genererar ordnad lista av steg, beroenden och successkriterier för att uppnå mål; kan vara statisk upfront eller uppdateras dynamiskt under körning.
**Dynamic Replanning** — Uppdaterar plan när verktygsresultat avviker från förväntan, nya constraints upptäcks eller tidigare steg visar sig omöjliga att slutföra.
**Reflection in Agents** — Agent utvärderar egna steg och korrigerar fel via metoder som Reflexion eller self-refine, ofta med separat kritik- eller verifieringspass.
**Self-Refinement Loop** — Generera → kritik → förbättra iterativt utan extern reward; modellen granskar eget utkast mot kriterier och skriver om tills kvaliteten accepteras.
**Verifier Agent** — Separat agent eller modell som kontrollerar korrekthet, format och policy compliance före att slutligt svar levereras till användaren eller externa system.
**Critic Agent** — Bedömer kvalitet på plan eller output och ger korrigerande feedback till executor; används i multi-agent-debatt och iterativ förbättring.
**Code Interpreter Agent** — Kör Python eller annan kod i sandbox för beräkning, dataanalys, filhantering och validering av hypoteser som LLM inte kan lösa rent textuellt.
**Browser Agent** — Navigerar webben via browser automation för att hämta information, fylla formulär eller interagera med SaaS utan dedikerade API:er.
**Computer Use Agent** — Styr GUI via mus och tangentbord för att interagera med skrivbordsapplikationer; kräver skärmbilder, OCR och noggrann action-sekvensering.
**Sandboxed Execution** — Isolerad miljö för agentkod och verktyg med begränsade permissions, nätverksfilter och resursgränser för att begränsa skada vid fel eller prompt injection.
**Tool Timeout** — Maximal tid per verktygsanrop för att undvika hängande agenter, oändliga väntan på externa API:er och att hela run förbrukar budget i ett enda steg.
**Max Iteration Limit** — Övre gräns på agent-loopar och verktygsanrop; förhindrar oändliga loopar när modellen upprepar samma misslyckade strategi.
**Action Budget** — Max antal verktygsanrop, tokens eller kostnad per uppgift; tvingar agenten att prioritera och avsluta när resurserna är slut.
**Human-in-the-Loop Agent** — Människa godkänner kritiska actions innan exekvering, särskilt vid muterande operationer, betalningar eller åtkomst till känslig data.
**Approval Gate** — Policy som kräver explicit bekräftelse för känsliga operationer; kan triggas per verktyg, riskklass eller beloppsgräns innan action körs.
**Autonomous vs Semi-Autonomous** — Full autonomi låter agenten slutföra hela uppgiften själv; semi-autonom innebär mänsklig övervakning, godkännande eller eskalering vid viktiga beslut.
**Agent Observability** — Logging av thoughts, tool calls, latens, fel och tokenanvändning för debugging, audit och förbättring av prompts och verktygsdesign.
**Trace Visualization** — UI som visar agentens resonemang, verktygskedja och mellanresultat i tidsordning så att utvecklare och användare kan följa beslutsvägar.
**OpenTelemetry for Agents** — Standardiserad tracing av agent-spans, verktygsanrop och beroenden; integreras med befintliga observability-stackar för produktion.
**Agent Evaluation** — Mäter task success rate, steg-effektivitet, kostnad, latens och säkerhet mot benchmarks eller golden tasks i kontrollerade miljöer.
**SWE-Bench Agent** — Benchmark för agenter som löser verkliga GitHub issues med kodändringar, tester och pull requests i riktiga repositories.
**WebArena Benchmark** — Simulerade webbuppgifter i realistiska miljöer för att utvärdera web-agenter på navigation, formulär och flerstegsinteraktion.
**Tool Use Accuracy** — Andel korrekt formaterade och semantiskt rätta verktygsanrop; mäter både schema-validering och om rätt verktyg valdes för uppgiften.
**Planning Error Recovery** — Agentens förmåga att återhämta sig från felaktiga plansteg genom omplanering, alternativa verktyg eller eskalering istället för att fastna.
**Ambiguity Handling** — Agenten ställer förtydligande frågor vid underspecificerade mål istället för att gissa; minskar felaktiga antaganden och dyra omvägar.
**Multi-Modal Agent** — Agent som hanterar text, bild, ljud och filer via verktyg eller inbyggd multimodal modell för att tolka och agera på rikare input.
**Retrieval Agent** — Specialiserad agent för sökning, filtrering och sammanställning av information från vektor-DB, webb eller interna kunskapsbaser.
**Research Agent** — Iterativ informationssökning, syntes, källkritik och citering; kombinerar retrieval, summarization och verifiering över flera källor.
**Coding Agent** — Skriver, kör tester, debuggar och itererar kod i repo-miljö med tillgång till filsystem, terminal och CI-liknande verktyg.
**DevOps Agent** — Automatiserar CI/CD, infrastrukturändringar, övervakning och incident response via API:er mot moln, Kubernetes och alert-system.
**Customer Support Agent** — Hanterar ärenden med CRM- och KB-integration, eskalering till människa och policy-styrda svar baserat på kundhistorik.
**Personal Assistant Agent** — Kalender, e-post, påminnelser och personliga preferenser med persistent minne och proaktiva förslag inom användarens godkända scope.
**Agent Persona** — Konfigurerad roll, ton, språk och capabilities som styr beteende, verktygsval och hur agenten kommunicerar med användaren.
**System Prompt for Agents** — Instruktioner om tillgängliga verktyg, säkerhetsregler, planeringsformat och output-struktur som definierar agentens grundbeteende.
**Structured Action Format** — XML, JSON eller YAML för actions som parsas deterministiskt av runtime istället för fri text som kan misstolkas vid exekvering.
**Parallel Tool Calls** — Flera oberoende verktyg körs samtidigt för lägre latens när resultat inte beror på varandra, t.ex. parallella API-sökningar.
**Sequential Tool Dependency** — Senare steg kräver output från tidigare verktyg; runtime måste respektera beroendegrafen och inte parallellisera beroende actions.
**Error Propagation in Agents** — Hur fel i verktyg rapporteras tillbaka till planner med statuskod, meddelande och kontext så att agenten kan retry eller omplanera.
**Retry Policy** — Regler för omförsök vid transienta verktygsfel med exponential backoff, max försök och jitter för att undvika thundering herd.
**Idempotent Tool Design** — Verktyg säkra att köra flera gånger utan oavsiktliga sidoeffekter; viktigt vid retry och osäker nätverksmiljö.
**Side-Effect Tracking** — Loggar muterande actions för audit, rollback och compliance; varje write operation kopplas till agent-run och användaridentitet.
**Agent Rollback** — Ångrar felaktiga ändringar efter misslyckad verifiering via compensating transactions, git revert eller snapshot-återställning.
**Checkpoint in Agent Run** — Sparar state, plan och verktygshistorik för att återuppta långa uppgifter efter timeout, crash eller mänsklig paus.
**Streaming Agent Output** — Streamar delvis plan, resonemang och resultat till användaren i realtid för bättre upplevelse vid långa flerstegskörningar.
**Background Agent** — Kör asynkront medan användaren gör annat; notifierar via webhook, e-post eller push när uppgiften är klar eller behöver input.
**Scheduled Agent Task** — Cron-liknande körning av agent på schema för rapporter, övervakning eller återkommande underhåll utan manuell trigger.
**Event-Triggered Agent** — Startar vid webhook eller systemevent, t.ex. nytt supportärende, alert eller PR, och agerar enligt fördefinierad playbook.
**Multi-Turn Tool Use** — Verktygsanrop över flera konversationsvarv med bibehållen state, minne och referens till tidigare resultat i samma session.
**Context Overflow in Agents** — Komprimering, summarization eller selektiv borttagning när historik överstiger kontextfönster utan att förlora kritiska mål och constraints.
**Conversation Summarization** — Periodisk sammanfattning av tidigare steg och verktygsresultat för att spara tokens och behålla relevant information i långa runs.
**Agent Prompt Injection Defense** — Filtrerar otillåten styrning från verktygsoutput och extern data innan den matas tillbaka till modellen som trusted instruction.
**Privilege Separation** — Olika verktyg och permissions per agent-roll så att en research-agent inte kan skriva till produktion utan eskalering.
**Least Privilege for Tools** — Minimal API-access per uppgift för att begränsa skada vid komprometterad prompt eller felaktig plan; scopes begränsas dynamiskt.
**Agent Guardrails** — Policylager som blockerar otillåtna actions före exekvering baserat på regex, klassificerare eller regelmotorer ovanpå LLM-beslut.
**Deterministic Tool Routing** — Regelbaserad routing istället för LLM-val vid kritiska paths, t.ex. alltid specifikt verktyg för betalning eller databaskörning.
**LLM Router Agent** — Liten modell klassificerar intent och väljer specialist-agent eller modellstorlek för kostnadseffektiv och träffsäker delegering.
**Mixture of Agents** — Flera agenter genererar parallella svar; aggregator eller domare kombinerar bästa delar eller väljer vinnande output.
**Debate Between Agents** — Agenter argumenterar mot varandra med motivering; domare eller konsensusprotokoll väljer slutligt svar efter flera ronder.
**Consensus Agent Protocol** — Kräver överenskommelse mellan agenter före action, särskilt vid högrisk-beslut eller när en agent är osäker.
**Agent Framework** — Ramverk som LangGraph, AutoGen eller CrewAI för att bygga agentflöden med state, verktyg, retries och observability inbyggt.
**LangGraph** — Graf-baserat orchestration för stateful LLM-agenter med cykler, checkpoints, human-in-the-loop och deterministiska transitions.
**AutoGen** — Multi-agent konversationsramverk med programmerbara agenter, gruppchatt och verktygsintegration för forskning och prototyping.
**CrewAI** — Roll-baserade agenter i team med delegerad arbetsfördelning, gemensamma mål och hierarkisk task assignment mellan roller.
**Agent SDK** — Bibliotek för tool definitions, runs, streaming och tracing, t.ex. OpenAI Agents SDK, som standardiserar agent-API:er.
**Computer-Using Agent Safety** — Risker när agent styr riktiga system; kräver sandboxes, action limits, skärmdumpsgranskning och human approval vid kritiska steg.
**Agent Cost Control** — Token-, API- och tidsbudget per run med varningar, hard stops och modellrouting för att hålla driftskostnad förutsägbar.
**Success Criteria Specification** — Maskinläsbara kriterier för när agenten ska stoppa, t.ex. verifierat testpass, dokumenterad källa eller användarbekräftelse.

## Multimodal AI

**Multimodal Model** — Modell som processar och genererar flera modaliteter som text, bild, ljud och video i gemensam arkitektur eller via kopplade encoders och decoders.
**Modality Alignment** — Träning så att representationer från olika modaliteter projiceras till delat semantiskt rum där matchande innehåll ligger nära oavsett ursprunglig modalitet.
**Vision-Language Model** — VLM: joint modellering av bild och text för förståelse, frågesvar, captioning och generering med cross-modal attention eller projicerade features.
**Image Encoder** — Neural encoder baserad på CNN eller ViT som mappar bild till vektorrepresentation eller patch-tokens för downstream fusion med text.
**Text Encoder in VLM** — Transformer-encoder för text som projiceras till samma embedding-rum som bildfeatures via linjära lager eller cross-attention.
**Projection Layer** — Linjärt eller MLP-lager som mappar en modalitets representation till gemensam embedding-dimension kompatibel med LLM eller fusion-modul.
**Contrastive Language-Image Pretraining** — CLIP: tränar bild- och textencoder med contrastive loss på par så att matchande bild-text ligger nära och icke-matchande separeras.
**CLIP Embedding Space** — Delat vektorrum där matchande bild-text-par ligger nära varandra och kan användas för retrieval, zero-shot klassificering och conditioning.
**Zero-Shot Image Classification** — Klassificering via textprompter utan task-specifik träning, med CLIP-liknande modeller som jämför bildembedding mot klass-textembeddings.
**Image-Text Matching** — Bedömer om bild och text beskriver samma innehåll via binary classifier eller cosine similarity i joint embedding space.
**Visual Question Answering** — VQA: svara på naturliga språkfrågor om bildinnehåll genom att kombinera visuella features med frågetokens i en enda modell.
**Image Captioning** — Generera textbeskrivning av bild via encoder-decoder, contrastive pretraining plus LM loss, eller VLM-generering med autoregressiv decode.
**Text-to-Image Generation** — Generera bild från textprompt via diffusion, autoregressiva token-modeller eller flow matching med text conditioning i hela pipelinen.
**Interleaved Image-Text Training** — Träning på sekvenser som växlar bild- och texttokens, likt webbdokument, för unified förståelse och generering i en modell.
**Any-to-Any Multimodal** — Modell som mappar godtycklig input-modalitet till godtycklig output, t.ex. text→video eller ljud→text, i en unified pipeline.
**Unified Tokenization** — Representerar bild, ljud och video som token-sekvenser i samma vocab eller embedding-tabell som texttokens för en enda transformer.
**Visual Tokenizer** — VQ-VAE, patch-embedding eller neural codec som diskretiserar bild till tokens som LLM kan behandla som sekvens.
**Audio Tokenizer** — Neural codec som EnCodec eller SoundStream som kodar ljud till diskreta tokens med hög kompression och acceptabel rekonstruktionskvalitet.
**Video Frame Encoding** — Samplear och encoderar bildrutor som sekvens, 3D-volym eller komprimerade latent frames för temporal modellering.
**Temporal Modeling in Video** — Fångar tidssamband via 3D-convolution, temporal attention över frames, state space models eller separata motion-moduler.
**Speech-to-Text** — ASR: transkriberar tal till text med encoder-decoder, CTC eller RNN-T; hanterar accent, brus och streaming i realtidssystem.
**Text-to-Speech** — TTS: syntetiserar naturligt tal från text via acoustic model, vocoder eller end-to-end neural TTS med prosodi och flera röster.
**Speech Encoder** — Modell som wav2vec eller Whisper encoder som extraherar fonetiska och semantiska features från rå ljudvågform eller mel-spektrogram.
**Whisper Architecture** — Encoder-decoder ASR tränad på massiv weakly supervised ljuddata med multitask objectives inklusive översättning och språkidentifiering.
**Audio-Language Model** — LLM som tar ljudtokens eller audio embeddings som input för förståelse, transkription, sammanfattning eller generering av tal och text.
**Omni Model** — En modell som hanterar text, bild och ljud in/ut i en pipeline med delade eller kopplade encoders och unified decoding.
**Cross-Modal Attention** — Attention mellan tokens från olika modaliteter i samma sekvens så att text kan referera till specifika bildpatches eller ljudsegment.
**Early Fusion** — Kombinerar modaliteter i tidiga lager av nätverket genom concatenation eller joint convolution innan högnivå semantik extraheras.
**Late Fusion** — Separata encoders per modalitet; fusion sker i slutet före task head via concatenation, averaging eller liten MLP.
**Middle Fusion** — Korsmodal interaktion i mellanliggande lager via cross-attention eller gating så att modaliteter påverkar varandra iterativt.
**Perceiver Multimodal** — Latent bottleneck som attendar över flera modaliteter oberoende av input-storlek; komprimerar till fix antal latenta tokens.
**Flamingo Architecture** — Fryst LLM med cross-attention till bildfeatures från vision encoder; endast adapter-lager tränas för effektiv multimodal kapacitet.
**LLaVA Architecture** — Projicerar CLIP- eller ViT-features till LLM via MLP connector; tränas med instruction tuning på bild-text-dialog.
**Q-Former** — Query transformer som extraherar fix antal visuella tokens till LLM via learnable queries och cross-attention mot bildfeatures.
**AnyRes Resolution Handling** — Delar högupplösta bilder i variabla patch-grid eller multi-scale crops så VLM kan hantera olika aspect ratios utan fix resize.
**Dynamic Resolution Input** — Variabel bildstorlek utan fix resize till kvadrat; bättre detaljbevarande via native resolution eller adaptiv patchificering.
**OCR in VLM** — Optisk teckenläsning integrerad i vision-language för dokument, skyltar och UI; kombinerar detection, recognition och språkförståelse.
**Document Understanding** — Layout, tabeller, rubriker och brödtext i PDF eller bild via multimodal modeller med spatial och semantisk reasoning.
**Chart and Diagram Reasoning** — Tolkning av grafer, diagram och visualiseringar i VLM; kräver numerisk och spatial förståelse bortom ren OCR.
**Spatial Reasoning in Vision** — Förståelse av position, storlek, djup och relationer mellan objekt i bilder för navigation, robotics och VQA.
**Grounding via Bounding Boxes** — Kopplar språk till regioner med koordinater, masker eller normaliserade boxar så modellen kan peka på specifika objekt.
**Referring Expression Comprehension** — Hitta objekt i bild givet naturlig språkbeskrivning, t.ex. 'personen till vänster med röd hatt'.
**Segmentation from Text Prompt** — Text-styrd segmentering via SAM plus CLIP eller unified models som genererar masker från fria textbeskrivningar.
**Multimodal In-Context Learning** — Few-shot med bild- och textexempel i prompten; modellen generaliserar uppgift utan gradientuppdatering vid inferens.
**Interleaved Generation** — Generera text och bild växelvis i samma session, t.ex. artikel med illustreringar eller stegvis visuell storytelling.
**Image Editing via Language** — Modifiera bild med instruktioner som inpainting, stilbyte eller object swap via diffusion, masker eller cross-attention manipulation.
**Video Captioning** — Beskriv video-innehåll i naturligt språk genom temporal aggregation av frame-features och autoregressiv eller template-baserad textgenerering.
**Video QA** — Frågesvar om händelser, objekt och temporal ordning i video; kräver minne över frames och förståelse av cause-effect.
**Multimodal RAG** — Hämtar bilder, diagram och text till gemensam kontext för LLM; indexerar media i vektor-DB med multimodal embeddings.
**Multimodal Embedding Search** — Sök bilder med text eller vice versa i delat index via cosine similarity i joint embedding space som CLIP.
**Modality Missing at Inference** — Hantera saknad modalitet vid inferens, t.ex. endast text eller endast bild, via training med modality dropout och robust fusion.
**Modality Dropout** — Slumpmässigt droppar en modalitet under träning för robusthet när vissa inputs saknas eller är noisy vid deployment.
**Alignment Loss** — Contrastive eller matching loss som synkar modalitetsrepresentationer så semantiskt lika innehåll ligger nära i embedding space.
**ITC Loss** — Image-Text Contrastive loss i CLIP-liknande träning; maximerar similarity för positiva par och minimerar för negatives i batch.
**ITM Loss** — Image-Text Matching binary classification loss som skiljer matchande par från hårda negatives med cross-encoder eller fusion head.
**LM Loss on Captions** — Autoregressiv caption loss som komplement till contrastive; tränar generativ kapacitet och finare språklig koppling till bild.
**Multimodal Pretraining Data** — Web-scale bild-text-par, videos, interleaved web documents och syntetisk data för bred coverage av koncept och domäner.
**Data Filtering for Multimodal** — Kvalitetsfilter för NSFW, blur, watermark och mismatch mellan bild och alt-text för renare träningsdistribution.
**Synthetic Multimodal Data** — Renderade scener, captions från LLM och simulerade miljöer för träning när mänsklig annotation är för dyr eller saknas.
**Audio-Visual Learning** — Joint modellering av ljud och video för lip sync, AVSR, event detection och multimodal fusion i media understanding.
**Lip Reading Model** — Predicerar tal från munrörelser i video utan ljud; används som komplement till ASR i bullriga miljöer.
**Music Generation Multimodal** — Generera musik från text, humör-beskrivning eller referensljud via diffusion, autoregression eller hybridmodeller.
**3D Understanding** — Point clouds, meshes och NeRF som input till AI-modeller för robotics, AR och scenförståelse i tre dimensioner.
**Point Cloud Encoder** — Network på 3D-punkter som PointNet, Point Transformer eller patch-baserade transformers för spatial feature extraction.
**NeRF Representation** — Neural Radiance Fields: implicit 3D-scen som kan renderas från godtyckliga vyer via volymetrisk ray marching.
**Gaussian Splatting** — Explicit 3D Gaussians för realtid rendering och redigering; används i 3D-gen AI och snabb novel view synthesis.
**Text-to-3D** — Generera 3D-objekt eller scener från text via diffusion, score distillation eller optimization över implicit eller explicit 3D-representation.
**Multimodal Safety** — Filter för skadligt bild- och textinnehåll, deepfake-risk och policybrott i genererat eller uppladdat multimodalt media.
**Deepfake Detection** — Klassificera AI-genererade ansikten, röster eller video mot autentiska via artifact detection, temporal inkonsistens och provenance.
**Provenance Metadata** — C2PA och liknande standarder som spårar ursprung, redigeringar och modell för genererat media i supply chain.
**Multimodal Benchmark** — Evalueringssuite för VQA, captioning, grounding och reasoning, t.ex. MME och MMMU, med standardiserade metrics.
**MMMU Benchmark** — Massive Multi-discipline Multimodal Understanding; expertfrågor med bilder från akademiska ämnen som testar djup reasoning.
**HallusionBench** — Testar hallucination i VLM på visuella påståenden genom att jämföra modellens svar mot faktiskt bildinnehåll.
**Multimodal CoT** — Chain-of-thought med visuella mellansteg, region reasoning eller markerade objekt för mer transparent multimodal problemlösning.
**Set-of-Mark Prompting** — Overlay numrerade markörer på bild för att guida VLM-attention till specifika regioner vid frågor och instruktioner.
**Image Token Budget** — Max antal visuella tokens per bild; påverkar tradeoff mellan detalj, latens och kostnad i VLM-inferens.
**Video Token Budget** — Subsampling av frames, temporal compression eller keyframe selection för långa videos inom LLM kontextgränser.
**Modality Adapter** — Lättviktsmodul som kopplar ny modalitet till fryst LLM utan full finetuning av basmodellen; ofta LoRA eller MLP.
**Unified Multimodal Decoder** — En decoder genererar tokens för alla modaliteter, t.ex. text och bildtokens i samma autoregressiva sekvens.
**Dual Encoder Retrieval** — Separata encoders per modalitet med shared contrastive space för effektiv bi-directional retrieval i stor skala.
**Cross-Modal Retrieval** — Hämta bild givet text eller text givet bild via nearest neighbor i joint embedding index.
**Multimodal Fine-Tuning** — Finjustera VLM på domänspecifika bild-text-uppgifter som medicin, retail eller industri med begränsad annotated data.
**Instruction Tuning for VLM** — Multimodal chat- och instruktionsdata för dialog om bilder, följ instruktioner och säker avvisning av olämpliga requests.
**Negative Image-Text Pairs** — Hard negatives i contrastive träning för skarpare decision boundaries mellan liknande men icke-matchande par.
**Resolution Extrapolation** — Inferens på högre upplösning än träningsdata via position interpolation, AnyRes eller multi-scale inference.
**Aspect Ratio Handling** — Bevarar bildproportioner vid patchificering genom dynamisk grid, padding eller crop istället för distorting resize.
**Color Space in Vision Models** — RGB vs YUV och normaliseringskonventioner påverkar pretrained features; mismatch vid deployment ger prestandaförlust.
**Multimodal Latency** — Vision encoder plus LLM decode; encoder och projektion är ofta flaskhals innan autoregressiv textgenerering startar.
**Visual Grounding** — Kopplar naturligt språk till specifika bildregioner via boxar, masker eller pekning för explainability och robot manipulation.
**Audio-Visual Speech Recognition** — Kombinerar lip-read video och ljud för robustare taligenkänning i brus via fusion i shared latent space.

## Diffusionsmodeller

**Diffusion Model** — Generativ modell som lär sig reversera gradvis brusning av data till sampling; tränas att predicera brus eller score vid varje tidssteg.
**Forward Diffusion Process** — Markovkedja som adderar Gaussisk brus till data över T tidssteg tills sample approximerar ren noise distribution.
**Reverse Diffusion Process** — Lär parametriserad modell att stegvis avlägsna brus och återskapa data genom att följa learned denoising transitions.
**Noise Schedule** — Varians β_t eller σ(t) över tidssteg styr hur snabbt signal försvinner och påverkar träningsstabilitet och sampling-kvalitet.
**DDPM** — Denoising Diffusion Probabilistic Models: grundläggande diskret tidssteg-diffusion med variational objective och Gaussian transitions.
**DDIM** — Deterministisk sampler med färre steg än DDPM utan extra träning; möjliggör snabbare inferens via non-Markovian process.
**Score Matching** — Lär ∇_x log p(x), score function, istället för explicit densitet; fundament för score-based generative modeling och SDE-formulering.
**Score-Based Generative Model** — SGM: SDE/ODE-perspektiv på diffusion och sampling med kontinuerlig tid och flexibla noise schedules.
**Denoising Score Matching** — Träna nätverk att predicera brus eller score givet noised input vid olika noise levels för effektiv generativ träning.
**Epsilon Prediction** — Modellen predicerar tillagt brus ε istället för x_0 direkt; numeriskt stabilt och standard i många U-Net diffusion implementationer.
**x0 Prediction** — Modellen predicerar ren data x_0 från noised sample; kan ge skarpare detaljer men mer känsligt vid höga noise levels.
**v-Prediction** — Interpolerad prediktionstarget som stabiliserar träning vid varierande SNR genom kombination av ε och x_0 i ett enhetligt target.
**Signal-to-Noise Ratio** — SNR: förhållande signal/brus vid tid t; styr loss-viktning, sampling och vilka tidssteg modellen fokuserar mest på.
**Variance Preserving Schedule** — VP-SDE: bevarar varians ungefär konstant under forward process; vanlig i DDPM och många text-to-image pipelines.
**Variance Exploding Schedule** — VE-SDE: varians växer under forward process; alternativ brusningsformulering med annan sampling och träningsdynamik.
**Latent Diffusion Model** — LDM: diffusion i komprimerat latent rum från VAE eller autoencoder för lägre compute och högre upplösning vid samma minne.
**VAE in Latent Diffusion** — Autoencoder komprimerar bild till latent z där diffusion sker; VAE decode i slutet återskapar pixlar från denoised latent.
**U-Net Denoiser** — U-Net-arkitektur som predicerar brus eller score med skip connections och multi-scale features för spatial denoising.
**Cross-Attention Conditioning** — Text eller embeddings conditionar U-Net via cross-attention layers, som i Stable Diffusion, för prompt-styrd generering.
**Classifier-Free Guidance** — CFG: kombinerar conditional och unconditional prediktion för skarpare samples och starkare prompt-följsamhet vid inferens.
**Guidance Scale** — Vikt på CFG-term; högre värde ger starkare prompt-följsamhet men kan over-saturate, artefakter och minska diversitet.
**Text Encoder for Diffusion** — CLIP, T5 eller OpenCLIP encoder som producerar text embeddings till U-Net cross-attention för semantisk conditioning.
**Timestep Embedding** — Sinusoidal eller learned embedding av diffusionsteg t injiceras i U-Net blocks för tidsberoende denoising beteende.
**Class-Conditional Diffusion** — Conditionar på klasslabel via embedding, adm eller null-class training för styrning utan fri textprompt.
**Inpainting Diffusion** — Maskerad region fylls i medan ok-maskerade områden conditionar via concatenated mask channels eller blended noise.
**Outpainting** — Genererar utökning utanför originalbildens kanter med seamless blending och conditioning på befintlig bildkontext.
**Image-to-Image Diffusion** — Startar från noised version av källbild med strength-parameter som styr hur mycket struktur som bevaras från input.
**ControlNet** — Auxiliary network injicerar spatial control som edges, depth, pose eller canny maps in i U-Net via zero-conv connections.
**T2I-Adapter** — Lättvikts adapter för strukturell kontroll utan full ControlNet; färre parametrar men mindre flexibel fine-grained styrning.
**LoRA for Diffusion** — Lågranks finjustering av U-Net och text encoder för stilar, koncept eller karaktärer med minimal extra lagring.
**DreamBooth** — Finjuterar på få bilder för att lära specifikt subjekt eller koncept med regularisering för att undvika overfitting och language drift.
**Textual Inversion** — Lär nytt 'word' embedding i text encoder från få exempel utan att uppdatera hela modellen; kompakt konceptrepresentation.
**IP-Adapter** — Image prompt adapter: conditionar generering på referensbild via decoupled cross-attention eller image embedding injection.
**Regional Prompting** — Olika textprompter för olika spatiala regioner via masker, attention manipulation eller compositional generation.
**Negative Prompt** — Text som modellen ska undvika via CFG unconditional branch; standard för att filtrera oönskade attribut i T2I.
**Sampler** — Algoritm som integrerar reverse process, t.ex. Euler, DPM++ eller Heun, med valbart antal steg och stokastisk eller deterministisk dynamik.
**Euler Discrete Sampler** — Enkel ODE-lösare med få steg; populär i Stable Diffusion för snabb inferens med acceptabel kvalitet vid rätt schedule.
**DPM-Solver** — Högre ordningens ODE-lösare för snabb sampling med få steg utan omträning; reducerar inferens från hundratals till ~20 steg.
**UniPC Sampler** — Unified predictor-corrector för effektiv diffusion sampling med bra tradeoff mellan hastighet och kvalitet vid låga stegantal.
**SDE vs ODE Sampling** — SDE ger stokastisk variation och diversitet; ODE mer deterministisk och reproducerbar given samma startnoise.
**Rectified Flow** — Lär direkt transport mellan noise och data längs raka paths; förenklar sampling och möjliggör få-stegs generering.
**Flow Matching** — Kontinuerlig normaliseringsflödes-träning utan simulering av full diffusion chain; effektiv alternativ formulering.
**Consistency Model** — En-stegs eller få-stegs generator tränad för konsistent denoising mapping från noise till data vid valfri tidssteg.
**Distillation for Diffusion** — Student modell lär sig få-stegs sampling från teacher genom matchning av output distribution eller trajectory.
**Progressive Distillation** — Iterativt halverar antal sampling-steg via distillation med bibehållen perceptuell kvalitet efter varje runda.
**Turbo/LCM Models** — Latent Consistency Models för realtid få-stegs generering med distillation från pretrained diffusion teacher.
**Video Diffusion** — 3D U-Net eller temporal attention för video-generering med frame consistency och motion modeling över tid.
**Temporal Attention in Video Diffusion** — Attention över frames eller latent temporal dimension för tidskoherens och smooth motion i genererad video.
**Audio Diffusion** — Diffusion i spektrogram, mel bins eller latent ljudrepresentation för musik, TTS och ljud-effekt generering.
**Diffusion Transformer** — DiT: transformer istället för U-Net som denoiser backbone med patchified latents och adaptive layer norm conditioning.
**Patchified Latent Input** — Delar latent bild i patchar som tokens till DiT; skalbar till hög upplösning med global self-attention.
**EDM Framework** — Elucidating Diffusion Models: unified formulering av schedules, preconditioning och loss för stabilare träning och sampling.
**Per-Resolution Training** — Multi-scale träning för bättre detalj och stabilitet genom att träna på flera upplösningar i samma modell.
**Min-SNR Weighting** — Viktning av loss baserat på SNR för balanserad träning över tidssteg och undvikande av dominerande höga noise levels.
**EMA Weights** — Exponential moving average av modellvikter för stabilare sampling och bättre generativ kvalitet vid inferens än raw weights.
**Diffusion Training Steps** — Antal tidssteg T under träning; kan skilja från inference steg när avancerade samplers subsamplar tidsaxeln.
**Inference Step Count** — Färre steg än träning via avancerade samplers; central tradeoff mellan kvalitet, latens och compute per bild.
**Mode Collapse in Diffusion** — Sällsynt men möjlig via dålig guidance, begränsad data diversity eller överstyrning som minskar output-variation.
**Exposure Bias in Diffusion** — Mindre relevant än autoregressiv generering; men schedule, sampler och VAE kan ändå ge systematiska artefakter.
**Safety Filter for Diffusion** — NSFW-klassificerare och policyfilter på prompt och output för att blockera olämpligt genererat innehåll.
**Watermarking Generated Images** — Osynlig vattenmärkning i diffusion outputs för spårbarhet och detektion av AI-genererat media.
**Causal Diffusion for Video** — Kausala temporal constraints för streaming video-gen utan access till framtida frames vid generering.
**Conditional Dropout** — Slumpmässigt droppar conditioning under träning för CFG-kompatibilitet och robust unconditional branch.
**Null Text Embedding** — Unconditional embedding för CFG unconditional branch; tränas via dropout av text conditioning.
**Prompt Weight Syntax** — Syntax med emphasis, parentheses eller numeriska vikter för att förstärka eller dämpa delar av prompt vid inferens.
**CLIP Score for Evaluation** — Cosine similarity CLIP(image, text) som kvalitetsproxy för prompt alignment utan mänsklig bedömning.
**FID for Diffusion** — Fréchet Inception Distance mellan genererade och riktiga bildfeatures; standard för overall generativ kvalitet.
**IS Inception Score** — Mått på bildkvalitet och diversitet via classifier entropy; mindre använt idag men historiskt viktigt för GAN och diffusion.
**Human Preference for T2I** — Elo-ranking eller pairwise comparison av bilder baserat på estetik och prompt-match från mänskliga annotatorer.
**GenEval Benchmark** — Objektantal, färg och spatial relation i T2I-evaluering med automatiska detektorer och strukturerade prompts.
**DPMSolver++** — Förbättrad DPM solver med bättre stabilitet vid låga steg och bred adoption i produktions inference pipelines.
**Karras Sigmas** — Noise schedule formulering optimerad för få-stegs sampling med bättre SNR-fördelning vid inferens.
**Sigma Schedule** — Explicit brusnivå σ(t) istället för β_t i vissa implementationer; underlättar sampler design och tuning.
**Diffusion Model Serving** — Batchad inferens med shared text encoding, parallel U-Net och VAE decode för throughput i produktion.
**VAE Decoder Artifacts** — Blur, ringing och color shift från latent decode; påverkar perceived skärpa oavsett U-Net kvalitet.
**Tiled VAE Decode** — Decode stora bilder i överlappande tiles för att passa GPU-minne utan att generera hela latents på en gång.
**MultiDiffusion** — Generera stora bilder genom överlappande diffusion-fönster med konsistent blending av partial denoising results.
**Semantic Diffusion Guidance** — Extra guidance från semantic segmentation, CLIP gradients eller classifier för finare semantisk kontroll.
**Prompt-to-Prompt** — Redigera bild genom att manipulera cross-attention maps mellan körningar med relaterade prompts.
**Attention Store in Diffusion** — Sparar cross- och self-attention maps för editing, interpretability och kontroll av vilka ord som påverkar vilka regioner.
**Null-Label Training** — Tränar unconditional branch parallellt med conditional via label eller text dropout för CFG vid inferens.
**Diffusion Prior** — Separat diffusion över CLIP embeddings, som DALL-E 2 prior, före decoder som genererar pixlar eller latents.
**Cascaded Super-Resolution** — Lågupplöst diffusion plus dedikerad upsampler eller super-res stage för högupplöst output i flerstegspipeline.
**Imagen Architecture** — Kaskad text-to-image med frozen T5, base diffusion och super-res stages för hög kvalitet vid stora modeller.
**Noise Offset Training** — Adderar liten konstant offset till brus för bättre kontrast, ljusvariation och undvikande av grå mediokritet.
**Offset Noise** — Low-frequency bruskomponent i träning som förbättrar global luminans och färgdiversitet i genererade bilder.
**Diffusion Model Quantization** — INT8 eller INT4 U-Net och text encoder för snabbare inferens med minimal perceptuell kvalitetsförlust via calibration.
**Diffusion Model Compilation** — torch.compile, TensorRT eller ONNX för optimerad inference pipeline med fusion och kernel selection.
**Stochastic Sampler** — Sampling som injicerar brus vid varje steg ger mer variation och kan förbättra diversitet jämfört med ren ODE.
**Deterministic Sampler** — Integrerar reverse ODE utan extra brus; reproducerbar given startseed och lämplig för reproducible benchmarks.

## Computer Vision

**Convolutional Neural Network** — CNN: hierarkiska conv-filter som fångar lokala spatiala mönster i bilder och bygger successivt högre nivå features via pooling.
**Vision Transformer** — ViT: behandlar bildpatchar som tokens med transformer-encoder; global self-attention ersätter conv för long-range dependencies.
**Image Classification** — Tilldelar hel bild en klasslabel från fördefinierat set via softmax över logits; grundläggande supervised vision task.
**Object Detection** — Hittar objekt med bounding boxes och klasslabels; one-stage eller two-stage pipelines med varierande latency-precision tradeoff.
**Instance Segmentation** — Pixelmask per objektinstans, inte bara klass per pixel; separerar individuella objekt av samma klass i scenen.
**Semantic Segmentation** — Klasslabel per pixel utan separation av instanser; används för scenförståelse, autonomous driving och medicinsk bild analys.
**Panoptic Segmentation** — Kombinerar semantic och instance segmentation i enhetlig representation med stuff och things i samma output.
**Keypoint Detection** — Predicerar anatomiska eller strukturella punkter som pose joints, facial landmarks eller objekt-specifika keypoints.
**Optical Flow** — Vektorfält som beskriver pixelrörelse mellan bildrutor; används i video analysis, action recognition och frame interpolation.
**Single-Shot Detector** — SSD: detekterar objekt i ett enda forward pass över multi-scale feature maps utan separat proposal stage.
**YOLO Architecture** — You Only Look Once: realtids object detection med grid-baserad prediktion och optimerad backbone för edge deployment.
**R-CNN Family** — Region-based detectors R-CNN, Fast R-CNN, Faster R-CNN med proposal network och ROI pooling för hög precision.
**Region Proposal Network** — RPN: genererar kandidat-regioner med objectness scores för two-stage detektorer som Faster R-CNN.
**Feature Pyramid Network** — FPN: multi-scale feature pyramid med top-down pathways för objekt i varierande storlekar i samma bild.
**Non-Maximum Suppression** — NMS: filtrerar överlappande boxes och behåller högsta score per objekt; kan utökas med soft-NMS eller class-aware varianter.
**IoU Metric** — Intersection over Union: overlap mellan predikterad och ground truth box; grund för mAP och träningsmatcher i detection.
**mAP** — Mean Average Precision: standardmetrik för object detection över IoU-trösklar och klasser; sammanfattar precision-recall.
**Anchor Boxes** — Fördefinierade box-skalaer och aspect ratios som detektor regresserar offset från; dominerade före anchor-free era.
**Anchor-Free Detection** — Predicerar objektcentrum, storlek och klass direkt utan fördefinierade anchors; enklare design i FCOS och CenterNet.
**Focal Loss** — Down-viktar lätta exempel i loss; adresserar extrem klassobalans i one-stage detectors som RetinaNet.
**RetinaNet** — One-stage detector med FPN och focal loss för hög precision utan two-stage latency; influerade modern detection.
**DETR** — Detection Transformer: set prediction med transformer encoder-decoder och bipartite matching utan NMS i träning.
**Hungarian Matching in DETR** — Optimal one-to-one matchning mellan pred och GT boxes via Hungarian algorithm för set loss utan duplicate predictions.
**Deformable DETR** — Deformable attention för effektiv multi-scale detection med färre queries och snabbare konvergens än vanilla DETR.
**Segment Anything Model** — SAM: promptbar grundmodell för segmentering via points, boxes eller mask hints; generaliserar brett med minimal finetuning.
**Mask R-CNN** — Utökar Faster R-CNN med mask head per region proposal för instance segmentation parallellt med box detection.
**U-Net for Segmentation** — Encoder-decoder med skip connections för pixelvis prediktion; standard i medicinsk bildsegmentering och biomedical CV.
**DeepLab** — Atrous convolution och ASPP för multi-scale semantic segmentation med large receptive field utan excessive downsampling.
**Atrous Convolution** — Dilated convolution: större receptive field utan downsampling genom att hoppa över pixlar i kernel application.
**Batch Normalization in CV** — Stabiliserar träning av djupa CNNs genom normalisering per channel; inferens använder running stats eller sync batch norm.
**Data Augmentation for Vision** — Random crop, flip, color jitter, mixup och cutout för generalisering och robusthet mot variation i deployment data.
**Mixup** — Linjär interpolation av bilder och labels som regularisering; mjukar decision boundaries och förbättrar calibration i vissa tasks.
**CutMix** — Klistrar in patch från en bild i annan med label-interpolation; tvingar modellen att använda delvis synliga objekt.
**AutoAugment** — Lärda augmentation-policies via sökning över transformation sequences; hittar domän-specifika augmentations automatiskt.
**Test-Time Augmentation** — TTA: aggregerar prediktioner över augmenterade views vid inferens för mer stabil och noggrann output.
**Transfer Learning in Vision** — Fine-tune ImageNet-pretrained backbone på downstream task med mindre data än träning från scratch kräver.
**ImageNet Pretraining** — Standard initiering med klassificering på 1k klasser; features generaliserar till detection, segmentation och retrieval.
**Self-Supervised Vision** — MAE, SimCLR, DINO: lär representationer utan manuella labels via pretext tasks på stora unlabeled dataset.
**Masked Autoencoder** — MAE: rekonstruera maskerade patchar med hög mask ratio; stark ViT-pretraining med effektiv encoder-only design.
**Contrastive Vision Learning** — SimCLR och MoCo: dra augmenterade views av samma bild nära och andra bilder långt i embedding space.
**DINO Self-Distillation** — Self-supervised ViT med teacher-student, centering och momentum för robust features utan labels.
**CLIP Vision Backbone** — ViT eller ResNet tränad med text contrastive; zero-shot capable och används som frozen encoder i många VLMs.
**Open-Vocabulary Detection** — Detektera klasser beskrivna med text, inte fix träningslista; möjliggör nya kategorier vid inferens utan omträning.
**Grounding DINO** — Open-set detector med text-conditioned queries som kopplar naturligt språk till detektion och grounding.
**OCR Pipeline** — Detektera textregioner, recognizera tecken och post-process med språkmodell för korrekt ordning och formatering.
**Scene Text Recognition** — STR: läsa text i naturliga scenbilder med varierande font, pose, blur och bakgrund utan begränsad lexikon.
**Document Layout Analysis** — Segmentera sidor i textblock, tabeller, figurer och rubriker för downstream parsing och RAG.
**Table Structure Recognition** — Extraherar rader, kolumner och cellinnehåll från tabellbilder för strukturerad data export.
**Face Recognition** — Embedding-baserad identifikation med metric learning och cosine distance i latent ansiktsrum.
**Face Verification** — Binärt beslut om två ansiktsbilder är samma person; används i autentisering med threshold på embedding distance.
**Liveness Detection** — Skiljer riktigt ansikte från foto, mask eller replay för anti-spoofing i biometrisk säkerhet.
**Pose Estimation** — 2D eller 3D kroppsledpositioner från bild eller video via heatmaps, regression eller transformer-based models.
**Action Recognition** — Klassificera aktivitet i videosekvens via temporal modeling, two-stream networks eller 3D convolutions.
**Tracking-by-Detection** — Detektera per frame och associera identiteter över tid med motion model och appearance matching.
**SORT Tracker** — Kalman filter för motion prediction plus Hungarian assignment för multi-object tracking med enkel pipeline.
**DeepSORT** — Lägger appearance embedding till SORT för robustare ID vid occlusion och kortvarig förlust av detektion.
**Re-Identification** — Re-ID: matcha samma person över olika kameror via metric learning på person embeddings.
**Stereo Vision** — Djup från stereo bildpar via disparity estimation och triangulation; klassisk geometri kombinerad med deep learning.
**Monocular Depth Estimation** — Predicera djupkarta från en bild med supervised labels, self-supervised cues eller foundation models.
**Structure from Motion** — SfM: rekonstruera 3D struktur och kameror från bildserier via feature matching och bundle adjustment.
**Visual SLAM** — Samtidig lokalisering och kartläggning från videoström för robot navigation och AR.
**Neural Radiance Fields** — NeRF: implicit scenrepresentation från multi-view bilder renderad via neural volume rendering.
**3D Object Detection** — Detektera objekt med 3D boxes i LiDAR, kamera eller fusion för autonomous driving perception stacks.
**Point Cloud Processing** — Deep learning på LiDAR punkter via voxelization, PointNet++ eller transformers för 3D perception.
**BEV Representation** — Bird's Eye View: top-down raster eller tensor för autonomous driving som unify sensor data i plan vy.
**Image Super-Resolution** — Rekonstruera högupplöst bild från lågupplöst input via CNN, GAN eller diffusion super-res modeller.
**Image Denoising** — Ta bort brus med CNN, transformer eller diffusion denoisers för fotografi och low-light enhancement.
**Style Transfer** — Överför stil från referensbild till innehållsbild via neural style, AdaIN eller attention-based methods.
**Domain Adaptation in Vision** — Generalisera från synthetic till real eller mellan dataset via adversarial alignment eller self-training.
**Adversarial Patch Attack** — Lokal patch optimerad för att foola detektor eller klassificerare oavsett var den placeras i scenen.
**Adversarial Robustness in CV** — Modellmotstånd mot små perturbationer av pixelvärden via adversarial training eller certified defenses.
**Model Explainability in Vision** — Grad-CAM, attention maps och saliency för att visualisera vilka regioner som driver prediktionen.
**Grad-CAM** — Gradient-viktad class activation map som highlight viktiga regioner för en given klass utan architectural changes.
**Saliency Map** — Heatmap över pixlar som påverkar prediktion mest; används för debugging och användarförtroende.
**Calibration in Vision Models** — Predikterade sannolikheter ska matcha faktisk accuracy per confidence bin; viktigt för säkerhetskritiska system.
**Open Images Dataset** — Storskalig multi-label detection och segmentation benchmark med miljontals annoterade bilder.
**COCO Dataset** — Common Objects in Context: standard för detection, segmentation, keypoints och captioning med 80 klasser.
**Image Resolution vs Accuracy** — Tradeoff mellan input-storlek, compute, minne och task-prestanda; högre res ger detalj men quadratic cost i attention.
**Real-Time Inference** — Optimering för video-FPS via quantization, TensorRT, pruning och mobile backbones som MobileNet.
**MobileNet Architecture** — Depthwise separable conv för effektiv mobil inferens med låg latency och begränsad strömförbrukning.
**EfficientNet** — Compound scaling av depth, width och resolution för optimal accuracy-efficiency tradeoff vid given compute budget.
**ONNX Export for Vision** — Portabel modell för cross-platform deployment mellan PyTorch, TensorFlow och inference engines.
**TensorRT Optimization** — NVIDIA inference engine med layer fusion, precision calibration och kernel autotuning för maximal throughput.
**Edge AI Vision** — On-device CV med begränsad compute, minne och ström; kräver komprimerade modeller och optimerad preprocessing.
**Synthetic Data for Vision** — Renderade eller simulerade bilder för träning med perfekt labels; minskar annotation cost men risk för sim2real gap.
**Sim2Real Gap** — Prestanda-förlust när modell tränad i sim deployas i verklighet p.g.a. domain shift i texture, lighting och sensor noise.
**Active Learning for Labeling** — Välj vilka bilder som ska annoteras för maximal modellförbättring via uncertainty eller diversity sampling.
**Weakly Supervised Detection** — Träna detektor med endast bildnivå-labels via MIL, CAM eller pseudo-labeling utan box annotation.
**Video Object Segmentation** — Segmentera specifikt objekt genom videosekvens med temporal propagation och first-frame mask eller text prompt.
**Temporal Consistency in Video** — Regularisering för stabil prediktion över frames via optical flow warping, temporal loss eller stateful models.

## Reinforcement Learning

**Reinforcement Learning** — Agent lär sig policy via trial-and-error med reward signal från miljö; optimerar långsiktig förväntad return istället för supervised labels.
**Markov Decision Process** — MDP: formalism med states, actions, transition probabilities, rewards och discount γ som modellerar sequential decision problems.
**Policy** — Mapping från state till action, deterministisk eller stokastisk; kan parametriseras av neuralt nätverk i deep RL.
**Value Function** — Förväntad kumulativ discounted reward från state V(s) eller state-action-par Q(s,a); central för bootstrapping och policy improvement.
**Q-Learning** — Off-policy TD-lärande av action-value function utan modell av miljön; uppdaterar Q mot max över nästa actions.
**SARSA** — On-policy TD som uppdaterar Q med faktiskt tagen nästa action; följer beteendet agenten faktiskt utför.
**Temporal Difference Learning** — TD: bootstrap från nästa states värde istället för full episod-return; lägre variance än Monte Carlo.
**Monte Carlo RL** — Uppdaterar från full episod-return utan bootstrapping; kräver episod termination och kan ha hög variance.
**Bellman Equation** — Rekursiv relation för värde V(s) = max_a [R(s,a) + γV(s')]; fundament för dynamic programming och RL algorithms.
**Bellman Optimality** — Optimal value satisfies self-consistency under best action; Bellman optimality equation definierar optimalt V* och Q*.
**Discount Factor** — γ ∈ [0,1] viktar framtida rewards; nära 1 betyder långsiktig planering, nära 0 myopisk optimering.
**Exploration vs Exploitation** — Balans mellan prova nya actions för information och utnyttja känd bra policy för omedelbar reward.
**Epsilon-Greedy** — Med sannolikhet ε slumpa action för exploration, annars greedy på Q eller policy för exploitation.
**UCB Exploration** — Upper Confidence Bound: välj action med högst osäkerhetsbonus baserat på besöksfrekvens och confidence interval.
**Thompson Sampling** — Bayesian exploration via sampling från posterior över rewards eller Q-värden; naturlig uncertainty-driven exploration.
**Policy Gradient** — Optimerar policy direkt genom gradient av förväntad reward med REINFORCE, actor-critic eller PPO.
**REINFORCE** — Monte Carlo policy gradient med full episod-return och optional baseline för variance reduction.
**Advantage Function** — A(s,a) = Q(s,a) - V(s) mäter relativ action-kvalitet jämfört med genomsnitt i state; central i actor-critic.
**Actor-Critic** — Actor uppdaterar policy medan critic uppskattar value för lower variance gradients än ren REINFORCE.
**A2C** — Advantage Actor-Critic: synkron parallell policy gradient med delade workers som samlar data batchvis.
**A3C** — Asynchronous Actor-Critic med parallella workers som uppdaterar globala weights asynkront utan replay buffer.
**PPO Clip Objective** — Clippad probability ratio r_t(θ) förhindrar för stora policy-uppdateringar och stabiliserar deep RL träning.
**Trust Region Policy Optimization** — TRPO: begränsar KL-divergens mellan gamla och nya policyn för monotonic improvement guarantees.
**Natural Policy Gradient** — Policy gradient med Fisher information metric för stabilare steg i policy space nära manifold.
**Soft Actor-Critic** — SAC: off-policy max-entropy RL för kontinuerliga actions med automatisk exploration via entropy bonus.
**Deterministic Policy Gradient** — DPG för kontinuerliga actions med deterministisk policy μ(s); grund för DDPG och TD3.
**Deep Q-Network** — DQN: Q-learning med neuralt nätverk, experience replay och target network för stabilitet på Atari och beyond.
**Experience Replay** — Buffer av transitions som bryter temporal korrelation vid träning och möjliggör återanvändning av data.
**Target Network** — Fryst kopia av Q-nätverk som uppdateras periodiskt eller soft update för stabila bootstrap targets.
**Double DQN** — Decouplar action selection och evaluation för att reducera Q overestimation bias i deep Q-learning.
**Dueling DQN** — Separerar value och advantage streams i Q-arkitektur för bättre generalisering när actions inte påverkar miljön.
**Prioritized Experience Replay** — Sample transitions proportionellt mot TD-error magnitude för effektivare lärande från surprising events.
**Rainbow DQN** — Kombinerar flera DQN-förbättringar som double, dueling, PER och distributional RL i en agent.
**Multi-Agent RL** — Flera agenter lär sig samtidigt i delad miljö med interaktion som gör miljön non-stationär från varje agents perspektiv.
**Cooperative MARL** — Agenter delar gemensamt mål och reward; kräver koordination och credit assignment mellan agenter.
**Competitive MARL** — Zero-sum eller adversarial interaktion mellan agenter som i self-play för spel och robust policy learning.
**Self-Play** — Agent tränar mot sig själv eller tidigare versioner; central i AlphaGo, OpenAI Five och emergent strategi.
**Curriculum in RL** — Progressivt svårare miljöer eller uppgifter under träning för stabilare lärande och undvikande av local optima.
**Reward Shaping** — Extra reward-termer som guidar lärande mot önskade beteenden; risk för reward hacking om shaping är feldesignad.
**Sparse Reward** — Reward endast vid mål; svår exploration som kräver intrinsic motivation, HER eller lång horisont planning.
**Dense Reward** — Frekvent feedback som underlättar lärande men kan biasera policy mot myopiska delmål istället för slutmål.
**Intrinsic Motivation** — Intern reward från curiosity, novelty eller prediction error för exploration när extrinsic reward är sparse.
**Curiosity-Driven Exploration** — Belönar besök av oväntade states via forward model prediction error eller random network distillation.
**Inverse RL** — Infererar reward function från expert demonstrations när true reward är okänd eller svår att specificera.
**Imitation Learning** — Lär policy direkt från expertdata utan explicit reward; snabb bootstrap men begränsad av expert quality.
**Behavioral Cloning** — Supervised learning på state→action från demonstrations; enkel men drabbas av covariate shift vid deployment.
**DAgger** — Dataset Aggregation: iterativt samlar corrections från expert på states policy faktiskt besöker under roll-out.
**GAIL** — Generative Adversarial Imitation Learning med discriminator som skiljer policy från expert och ger learned reward.
**Offline RL** — Lär från statisk dataset utan online miljöinteraktion; måste hantera distributional shift och extrapolation error.
**Batch RL** — Synonym till offline RL; batch av transitions från logging policy utan möjlighet att samla ny data under träning.
**Conservative Q-Learning** — CQL: straffar Q-värden på out-of-distribution actions för att undvika overestimation i offline setting.
**Model-Based RL** — Lär miljömodell och planerar med den via Dyna, MBPO, MuZero eller MPC för högre sample efficiency.
**MuZero** — Lär modell implicit för planning och MCTS utan explicit rekonstruktion av full observation state.
**AlphaZero** — Self-play plus MCTS plus neural policy/value network för superhuman brädspel utan mänsklig data.
**Monte Carlo Tree Search** — MCTS: simulerar framtida spelträd med selection, expansion, simulation och backprop för action selection.
**UCB1 in MCTS** — Selection via upper confidence bound i sökträd balanserar exploration av nya noder och exploitation av lovande grenar.
**Partially Observable MDP** — POMDP: agent ser inte full state; behöver belief state, observation history eller recurrent policy.
**Recurrent Policy** — RNN eller Transformer i policy för att hantera partial observability och minne över tid i POMDP.
**Continuous Action Space** — Actions i R^n som joint torques eller steering; kräver policy som outputtar realvektorer och ofta Gaussian noise.
**Action Discretization** — Kvantiserar kontinuerliga actions till finite set för Q-learning; enkel men kan förlora precision.
**Sim-to-Real Transfer** — Policy tränad i sim deployas på riktig robot eller hardware med domain randomization och system identification.
**Domain Randomization** — Randomiserar sim-parametrar som friction, lighting och mass under träning för robust real-world transfer.
**RLHF Connection** — RL med learned reward model från mänskliga preferenser; koppling mellan classical RL och LLM alignment.
**Reward Model Overfitting** — RM som inte generaliserar till OOD policy outputs ger missvisande reward och policy collapse vid RL fine-tuning.
**KL Regularization in RL** — Straffar avvikelse från referenspolicy i RL fine-tuning för att behålla capabilities och undvika reward hacking.
**Constrained RL** — Maximera reward under säkerhets-, resurs- eller fairness constraints via Lagrangian methods eller CMDPs.
**Safe RL** — Undviker farliga states och actions under lärande och deployment via shields, constraints och risk-sensitive objectives.
**Multi-Objective RL** — Pareto-optimal policy över flera reward-komponenter; användaren eller scalarization väljer tradeoff vid deployment.
**Hierarchical RL** — Options och skills på olika tidsskalor för komplexa uppgifter med temporal abstraction och subgoal discovery.
**Options Framework** — Temporally extended actions macro-actions med initiering, policy och termination conditions i hierarkisk RL.
**Goal-Conditioned RL** — Policy conditionad på målstate eller goal embedding; generaliserar över tasks i samma miljö.
**HER Hindsight Experience Replay** — Behandlar uppnådda states som surrogate goals retroactively för att lära från misslyckade episoder i sparse reward.
**Policy Distillation** — Komprimera ensemble eller stor teacher policy till mindre student för deployment med bibehållen ungefärlig beteende.
**World Model** — Predicerar nästa observation och reward; används för planning i latent space som i Dreamer och MBRL.
**DreamerV3** — Model-based RL med latent imagination i learned world model för hög sample efficiency på diverse domains.
**Sample Efficiency** — Antal miljöinteraktioner som krävs för given prestanda; kritiskt i robotics och dyra real-world miljöer.
**Regret in RL** — Kumulativ skillnad mot optimal policy över tid; teoretiskt mått på online learning performance.
**On-Policy vs Off-Policy** — On-policy tränar på data från aktuell policy; off-policy återanvänder gammal data via replay med importance sampling.
**Importance Sampling in RL** — Korrigerar off-policy data med likelihood ratio mellan behavior och target policy för unbiased gradient estimates.
**Generalized Advantage Estimation** — GAE: bias-variance tradeoff i advantage estimation via exponential averaging av TD residuals.
**Entropy Bonus** — Regularisering som uppmuntrar utforskning i policy gradient genom att maximera policy entropy i objective.
**Non-Stationarity in MARL** — Andra agents policy ändras under träning så miljön blir icke-stationär från varje agents perspektiv.
**Centralized Training Decentralized Execution** — CTDE: global info och joint critic vid träning men lokal decentraliserad policy vid körning.
**RL Environment API** — Gymnasium/Gym interface reset(), step(action) → observation, reward, terminated, truncated, info för standardiserad integration.
**Partial Episode Bootstrapping** — Truncated episodes bootstrap från value estimate vid timeout istället för noll return vid icke-terminal truncation.
**Reward Normalization** — Skalar eller standardiserar rewards för stabilare value learning och konsekvent hyperparameter känslighet.
**Observation Normalization** — Normaliserar state features till zero mean unit variance baserat på running statistics från miljö sampling.
**Simulated Benchmark** — MuJoCo, Atari, Procgen och DM Control för standardiserad RL-evaluering och algoritmjämförelse.
**Exploration Bonus** — Extra reward för att besöka nya states eller reducera uncertainty; mitigerar sparse-reward-problem i stora miljöer.
**Exploitation Policy** — Policy som maximalt utnyttjar kända högbelönade actions; optimal vid deployment men insufficient alone under träning.

## Utvärdering & Benchmarks

**Benchmark Dataset** — Standardiserad testsuite med fasta uppgifter och mätvärden som gör modellutvärdering jämförbar mellan labb, tidpunkter och leverantörer. Gemensamma protokoll minskar risken att resultat bara speglar ad hoc-promptdesign eller dataläckage.
**Leaderboard** — Publikt rankat resultat på en benchmark där modeller listas efter score, ofta med metadata om setup och datum. Driver reproducerbarhet, tävling och snabb överblick, men kan också förstärka överoptimering mot en enskild metric.
**Held-Out Test Set** — Data som strikt hålls borta från träning, validering och hyperparameter-sökning tills slutlig utvärdering. Syftet är att ge en opartisk uppskattning av generalisering utan att modellen indirekt "sett" testfrågorna under utveckling.
**Validation Set** — Separat delmängd som används under utveckling för att välja hyperparametrar, early stopping och modellvarianter. Får aldrig blandas ihop med slutligt testset, annars blir den rapporterade testprestandan optimistiskt snedvriden.
**Cross-Validation Score** — Aggregerad prestanda över flera k-fold-splits där varje fold roterar som validering medan övriga används för träning. Ger stabilare uppskattning än en enda split, särskilt när datamängden är begränsad.
**Statistical Significance Test** — Formell analys (t-test, bootstrap, permutationstest) för att avgöra om skillnaden mellan två modeller sannolikt beror på slump eller verklig förbättring. Utan detta riskerar små dataset att ge missvisande "vinster" på brus.
**Bootstrap Confidence Interval** — Resampling av testprediktioner eller körningar för att uppskatta osäkerhetsintervall kring ett medelvärde eller skillnad. Särskilt användbart när testmängden är liten eller fördelningen av fel inte är normal.
**Standard Error of Mean** — SEM mäter hur mycket medelvärdet av en metric skulle variera över upprepade eval-körningar med samma setup. Hjälper till att skilja stabil förbättring från fluktuationer i sampling eller decoding.
**Effect Size** — Kvantifierar hur stor skillnaden är i praktiken, oberoende av sample size, till exempel via Cohen's d eller odds ratio. Ett litet p-värde säger ingenting om skillnaden är meningsfull i produktion.
**Multiple Comparison Correction** — Justering (Bonferroni, Benjamini–Hochberg FDR) när många hypoteser eller benchmarks testas samtidigt. Utan korrektion ökar risken att någon metric "vinner" av ren slumpmassa.
**Human Evaluation Protocol** — Dokumenterad process med blind ranking, tydliga rubrics, inter-rater agreement och kvalitetskontroll av annotatörer. Mänsklig bedömning fångar nytta och ton som automatiska metrics ofta missar, men kräver disciplin för att vara reproducerbar.
**Elo Rating for Models** — Parvis jämförelse av modelloutputs aggregeras till en global ranking via Elo-liknande uppdatering, som i Chatbot Arena. Skalar till många modeller men är känslig för promptval, domare och population av användare.
**LLM-as-Judge Evaluation** — Automatisk bedömning där en stark LLM rankar, betygsätter eller jämför svar enligt rubric. Mycket skalbart och billigare än människor, men domaren har egna bias (längd, stil, self-preference).
**Reference-Based Metric** — Kvalitetsmått som jämför modellens output mot en gold reference, till exempel BLEU, ROUGE eller chrF. Enkelt att automatisera men straffar giltiga omskrivningar som inte matchar referensen ordagrant.
**Reference-Free Metric** — Kvalitetsmått utan gold-svar, till exempel perplexity, LLM-judge eller MAUVE. Nödvändigt när det finns många korrekta svar eller när referenser saknas, men svårare att tolka och mer bias-känsligt.
**BLEU Score** — N-gram precision mot en eller flera referenser; historiskt dominerande metric i maskinöversättning. Hög BLEU betyder lexikal overlap men korrelerar inte alltid med mänsklig flyt eller mening.
**ROUGE Score** — Recall-orienterad n-gram overlap mot referens; vanlig i sammanfattning där täckning av viktiga fakta prioriteras. ROUGE-L fångar delvis sekvensmatchning men missar semantisk ekvivalens.
**METEOR** — MT-metric som väger in synonymer, stemming och ordning utöver rå n-gram overlap. Var designad för bättre korrelation med mänskliga bedömningar än ren BLEU, särskilt vid parafras.
**chrF** — Character n-gram F-score som är robust för morfologiskt rika språk och stavningsvariation. Används ofta som komplement till ordnivå-metrics när tokenisering skiljer sig mellan system.
**BERTScore** — Embedding-likhet mellan tokens i candidate och reference via contextualiserade representationer. Fångar semantisk närhet bättre än n-gram men beror på val av embedding-modell och kan vara långsam.
**COMET** — Neural MT-metric tränad att förutsäga mänskliga kvalitetsbedömningar från source, hypothesis och optional reference. Ofta stark korrelation med human ratings men kräver att man litar på träningsdistributionen.
**Perplexity Evaluation** — Exponentierad genomsnittlig cross-entropy på en testkorpus; lägre perplexity innebär bättre sannolikhetsmodell av text. Användbart för språkmodeller men korrelerar inte alltid med downstream-uppgiftskvalitet.
**Bits Per Byte** — Perplexity normaliserad per byte i stället för per token, vilket gör jämförelse mer rättvis över olika tokenizers och språk. Vanligt i komprimerings- och foundation model-rapportering.
**Exact Match** — EM: andel exakt matchande svar efter normalisering; vanligt i extractive QA och enkla klassificeringsuppgifter. Strikt och lätt att tolka men ignorerar semantiskt korrekta svar med annan formulering.
**F1 in QA** — Token-overlap F1 mellan predikterat och gold-svar; balanserar precision och recall på ordnivå. Mer tolerant än EM men kan ge högt betyg för delvis rätt svar som saknar nyckelinformation.
**SQuAD Benchmark** — Reading comprehension QA på Wikipedia-passager med korta svarsspann. Grundbult i NLP-historien men nu delvis mättad; många moderna modeller närmar sig taket på v1.1.
**MMLU** — Massive Multitask Language Understanding: 57 ämnen med flervalsfrågor som testar bred kunskap. Standard för "general knowledge" men känslig för promptformat, few-shot-val och träningsdatacontamination.
**HellaSwag** — Commonsense sentence completion med adversarial distractors designade för att lura ytlig mönstermatchning. Testar om modellen förstår vardagssituationer bortom enkel n-gram-plausibilitet.
**ARC Challenge** — Science QA som kräver multi-step resonemang och inte bara retrieval av fakta. Challenge-split skiljer starka modeller tydligare än enklare ARC Easy.
**GSM8K** — Grade school math word problems som testar numeriskt resonemang och steg-för-steg-lösning. Ofta evaluerad med och utan chain-of-thought; fel i sista steget ger noll trots delvis korrekt resonemang.
**MATH Benchmark** — Competition-level matematik med långa lösningar och flera delproblem. Kräver verifierare eller LLM-judge med hög precision; liten förbättring här indikerar starkare formellt resonemang.
**HumanEval** — Python kodgenerering där korrekthet verifieras med unit tests på dolda testfall. Pass@k är standardmetric; benchmark mäter syntes men inte kodkvalitet, säkerhet eller underhållbarhet.
**MBPP** — Mostly Basic Python Problems: enklare kod-syntesuppgifter än HumanEval, ofta med fler korta program. Bra för att mäta grundläggande programmeringsförmåga hos mindre modeller.
**SWE-Bench** — Verkliga GitHub-issues där en agent måste navigera repo, patcha kod och passera tester. Närmare produktions-ML-engineering än isolerade kodsnippets men dyr och svår att reproducera exakt.
**BigCodeBench** — Diverse programming tasks med library-aware evaluation som testar användning av API:er och paket. Mäter om modellen kan skriva praktisk kod, inte bara algoritmiska toy-problem.
**TruthfulQA** — Mäter sanningsenlighet mot vanliga missuppfattningar och "trick"-frågor där människor ofta svarar fel. Viktig för att upptäcka hallucinationer och imitativa men falska svar.
**ToxiGen** — Benchmark för toxisk generering och bias mot specifika grupper i genererad text. Används både för att mäta modellrisk och för att träna moderationsklassificerare.
**BBQ Benchmark** — Bias Benchmark for QA med frågor designade för att avslöja stereotyp bias i ambiguous respektive disambiguated kontext. Rapportering per undersgrupp krävs för meningsfull tolkning.
**WinoBias** — Coreference resolution med gender–career bias där pronomen ska kopplas rätt trots stereotypa associationer. Avslöjar om modellen faller tillbaka på sociala fördomar i språkförståelse.
**HELM Holistic Evaluation** — Brett eval-ramverk som kombinerar många scenarios, metrics, calibration och effektivitet i en enhetlig rapport. Syftar till att undvika cherry-picking av en enda benchmark.
**AlpacaEval** — Win rate mot en reference-modell via automated judge, ofta GPT-4-liknande. Populärt för instruction-tuned modeller men domare och reference dominerar den absoluta skalan.
**MT-Bench** — Multi-turn conversation quality bedömd med stark LLM-judge på flera ämneskategorier. Testar om modellen håller kontext och kvalitet över flera turer, inte bara enstaka svar.
**Arena-Hard** — Svårare prompt-subset från arena-trafik för att skilja top-modeller när enklare frågor ger tie. Minskar mättnad men kan överrepresentera vissa användartyper.
**Needle in a Haystack** — NIAH: modellen ska hitta en specifik "nål" (fakta) inbäddad i mycket lång distraktionstext. Standardtest för effektiv long-context retrieval och position bias i KV-cache.
**LongBench** — Suite av uppgifter för long-context förståelse, sammanfattning och reasoning över långa dokument. Kompletterar syntetiska NIAH med mer realistiska dokumentstrukturer.
**RULER Benchmark** — Synthetic long-context tasks med kontrollerad svårighet, kontextlängd och nålposition. Gör det möjligt att isolera om fel beror på minne, retrieval eller resonemang.
**Pass@k Metric** — Sannolikhet att minst ett av k oberoende samples löser uppgiften; standard i kodgenerering. Kräver tillräckligt många samples för stabil skattning, särskilt när bas-success är låg.
**maj@k** — Majority vote över k samples vid eval, till exempel vid flera CoT-svar eller klassificering. Kan höja accuracy men maskerar osäkerhet om modellen är konsekvent fel på samma sätt.
**Calibration Error** — Mäter hur väl predikterad konfidens matchar faktisk accuracy, ofta via ECE eller reliability diagrams. Viktigt när modellen används för selektiv prediction eller riskstyrning.
**Expected Calibration Error** — Viktad medel absolut avvikelse mellan konfidens och accuracy per konfidens-bin. ECE är standard i klassificering och växande also i LLM confidence-rapportering.
**Brier Score** — Mean squared error mellan probabilistisk prediktion och faktiskt binärt outcome. Straffar både felaktiga prediktioner och överdriven säkerhet; lägre är bättre.
**Selective Prediction** — Modellen får avstå när konfidens är låg; mäts tradeoff mellan coverage (andel besvarade) och accuracy på besvarade. Centralt i medicinska och juridiska tillämpningar där fel är dyra.
**Abstention Evaluation** — Mäter kvalitet när modellen explicit säger "vet inte" eller avstår, inte bara accuracy på svar den gav. Bra abstention ska maximera precision på besvarade utan att ducka för många enkla frågor.
**Adversarial Evaluation** — Systematisk test av robusthet mot adversarial inputs, trigger prompts eller perturberade exempel. Avslöjar fragilitet som clean benchmark-score döljer.
**Dynamic Benchmark** — Kontinuerligt uppdaterade frågor eller held-out pools för att minska träningsdatacontamination och memorering. Kräver versionshantering och kan göra historisk jämförelse svårare.
**Contamination Detection** — n-gram overlap, embedding-sökning eller membership inference mot misstänkt träningskorpus. Ingen metod är perfekt; kombination och manuell audit rekommenderas.
**Data Leakage Audit** — Systematisk kontroll att test-, val- och benchmark-exempel inte fanns i träning eller pretraining, direkt eller nästan identiskt. Kritiskt innan man publicerar state-of-the-art-claims.
**Benchmark Saturation** — När top-modeller närmar sig taket på en metric tappar benchmark discriminativ kraft och små skillnader blir brus. Signal att behöva svårare uppgifter eller finare mänsklig eval.
**Goodhart's Law in Benchmarks** — När en metric blir mål optimeras den ofta på bekostnad av verklig nytta som metric inte fångar. Exempel: högre leaderboard-score men sämre användarupplevelse i produkt.
**Task Contamination** — Uppgiftstyp, mall eller specifika exempel överlappar träningsdata så att hög score delvis speglar memorering. Skiljs från generell domänkunskap genom dedup och canary-set.
**Prompt Sensitivity Analysis** — Kartlägger hur små promptändringar (ordning, rubric, språk) påverkar score. LLM-eval utan denna analys riskerar att jämföra prompt-lucka snarare än modellkapacitet.
**Format Sensitivity** — Prestanda beror starkt på output-format (JSON vs fri text, markdown-tabeller, kodblock). Rapportera alltid formatkrav; annars är cross-paper-jämförelse missvisande.
**Position Bias in Eval** — Judge eller modell favoriserar svar i viss ordning (A före B) i parvis jämförelse. Mitigeras med position swapping och medel över båda ordningar.
**Self-Preference Bias** — Modell som judge favoriserar egna genererade svar eller samma familj/stil. Vanligt i LLM-as-judge; mitigera med anonymisering, extern judge eller människa.
**Length Bias in Eval** — Längre svar bedöms högre oavsett innehållskvalitet, särskilt av LLM-domare. Normalisera, begränsa max tokens eller inkludera length-controlled human eval.
**Regression Testing for Models** — Kör samma eval suite vid varje modellrelease för att fånga regressioner på viktiga uppgifter. Motsvarar enhetstester men för modellbeteende över tid.
**Canary Eval Set** — Liten hemlig testsuite som aldrig publiceras eller läcker till träning; används som tidig varning för överanpassning. Kompletterar publika benchmarks som lättare kontamineras.
**Shadow Deployment Eval** — Ny modell kör parallellt mot produktion på live traffic utan att påverga användare; outputs loggas för jämförelse. Bryggar gap mellan offline benchmark och verklig användning.
**A/B Test for Models** — Randomiserad trafik till modell A vs B med affärs- och kvalitetsmetrics (CTR, retention, thumbs). Guldstandard för produktpåverkan men kräver tillräcklig trafik och etisk granskning.
**Online Metric vs Offline Metric** — Korrelation mellan lab benchmark och produkt-KPI (engagemang, supportärenden, konvertering). Låg korrelation betyder att offline-optimering inte nödvändigtvis hjälper användare.
**Cost-Adjusted Benchmark** — Normaliserar score per inference-kostnad, latens eller energi så att dyrare modeller jämförs rättvist. Viktigt när "bästa" modellen inte är deploybar ekonomiskt.
**Energy Efficiency Metric** — Joules per query, per token eller per tränad token för att jämföra hållbarhet. Kompletterar accuracy i beslut om modellstorlek och hårdvaruval.
**Latency SLA Evaluation** — Andel requests under p95/p99 latens-budget vid given load och kontextlängd. En modell kan ha hög accuracy men faila produktion om tail-latens spränger SLA.
**Fairness Metric** — Equalized odds, demographic parity, calibration per grupp och relaterade kriterier över skyddade attribut. Val av metric är normativt; flera bör rapporteras disaggregerat.
**Disaggregated Evaluation** — Rapporterar metrics per subgrupp (språk, domän, svårighetsgrad, region) i stället för bara macro average. Macro-hög score kan dölja katastrofal prestanda för minoritetsgrupper.
**Error Analysis** — Manuell eller halvautomatisk kategorisering av feltyper efter eval (hallucination, reasoning, format, kunskap). Omvandlar en siffra till actionable förbättringslista för träning och produkt.
**Confusion Matrix Analysis** — Systematisk genomgång av vanliga felklasser och systematiska förvirringar mellan kategorier. Särskilt viktigt i klassificering och moderation där vissa fel är dyrare än andra.
**Qualitative Eval** — Expert- eller användargranskning av representative samples kompletterar automatiska metrics. Fångar tonalitet, säkerhet och nytta som siffror missar; kräver strukturerade protokoll.
**Red Team Eval Report** — Strukturerad rapport över genomförda säkerhetstester, exploits, severity och åtgärder. Skiljer seriös säkerhetsdue diligence från ad hoc "vi provade några prompts".
**Benchmark Versioning** — Versionera dataset när frågor uppdateras, tas bort eller svårighet ändras; dokumentera changelog. Utan version blir tidsserier av leaderboard-resultat omöjliga att tolka.
**Eval Harness** — Standardiserat ramverk (lm-eval, Eleuther harness) som kör samma task definitions, prompts och metrics reproducerbart. Minskar implementationsskillnader mellan papers och produkter.
**lm-eval Integration** — Kör många benchmarks med en CLI och config-filer; de facto standard i open-source LLM-eval. Underlättar jämförelse om samma task-version och model adapter används.
**Few-Shot Eval Protocol** — Fix antal demonstrations-exempel per task i prompt, med exakt samma exempel och ordning för alla modeller. Små avvikelser i exemplen kan ändra ranking mer än modellskillnad.
**Zero-Shot Eval Protocol** — Ingen demonstration i prompt; endast task instruction. Mäter instruktionsföljning och inbyggd kunskap utan ICL; standard för många public leaderboard-körningar.
**Chain-of-Thought Eval** — Jämför prestanda med och utan CoT på reasoning-uppgifter. Rapportera båda och om CoT aktiveras implicit via prompt, annars blir jämförelser mellan modeller snedvridna.
**Self-Consistency Eval** — Majority vote eller aggregering över flera reasoning paths sampled från samma modell. Ofta höjer score på GSM8K/MATH men ökar inference-kostnad linjärt med antal samples.
**Verifier-Based Eval** — Extern checker (kompilator, CAS, unit tests, formell verifierare) avgör korrekthet objektivt. Prefereras när gold reference är entydig; minskar judge-bias i matematik och kod.
**Human Likert Scale** — Bedömning på till exempel 1–5 skala med definierade ankarpunkter per nivå. Kräver kalibrering av annotatörer och rapportering av inter-rater reliability (Cohen's kappa, Krippendorff).
**Pairwise Preference Eval** — Människa eller judge väljer bästa av två outputs utan absolut betyg. Grunden för Elo och RLHF; känslig för position, length och tie-breaking policy.
**Win Rate Metric** — Andel gånger modell A slår B (eller reference) i parvis jämförelse, ofta med ties exkluderade eller räknade som 0.5. Enkel att kommunicera men döljer magnitude of skillnad och segment-specifik prestanda.

## MLOps

**MLOps** — Praxis och verktyg för att operationalisera ML från experiment till produktion: CI/CD, monitoring, governance och reproducerbarhet. Målet är att modeller ska vara spårbara, säkra att uppdatera och mätbara i verklig drift, inte bara i notebook.
**Model Registry** — Central catalog över modellversioner med metadata, lineage, godkännandestatus och stage (staging/production). Gör rollback, audit och team-samarbete möjligt utan ad hoc filnamn på S3.
**Experiment Tracking** — Loggar hyperparametrar, metrics, kodcommit och artifacts per träningskörning (MLflow, W&B). Utan spårbarhet går det inte att reproducera "vilken körning som faktiskt var bäst" veckor senare.
**MLflow** — Open-source plattform för experiments, model registry, packaging och deployment-integration. Vanlig i on-prem och moln; styrker när man vill ha en enda källa för runs och modellpromotion.
**Weights & Biases** — W&B: molnbaserat experiment tracking, hyperparameter-sweep och team-dashboards. Populärt för deep learning med rik visualisering och delning av runs mellan forskare och ingenjörer.
**Feature Store** — Centraliserad lagring av features med samma definition i offline träning och online serving. Löser training-serving skew och duplicerad feature-logik mellan data scientists och produktionsteam.
**Training Pipeline** — Automatiserad orkestrering av datainhämtning, validering, träning, eval och registrering av modell. Körs på schedule eller trigger; ersätter manuella "kör notebook på fredag"-processer.
**Inference Pipeline** — Produktionsflöde för preprocessing, model forward pass, postprocessing och logging per request eller batch. Måste matcha träningspreprocessing exakt och hantera fel, timeout och fallback.
**CI/CD for ML** — Kontinuerlig integration och deployment med automatiska tester, datavalidering och model validation gates vid varje ändring. Skiljer sig från vanlig CI/CD genom icke-deterministiska metrics och stora artifacts.
**Model Validation Gate** — Automatiska kvalitetskontroller (accuracy, fairness, latency, safety) som måste passera innan deploy till staging/production. Blockerar regressioner som enhetstester inte fångar.
**Data Validation** — Great Expectations-liknande checks på schema, distribution, null-rate och värdeintervall vid varje pipeline-steg. Fångar brutna upstream-ETL innan dålig data tränar eller serverar fel predictions.
**Schema Drift Detection** — Larm när indatafält, typer eller kardinalitet ändras oväntat jämfört med träningsbaseline. Vanligt orsak till plötslig produktionsdegradering efter API- eller databasändring.
**Data Versioning** — DVC, Git LFS eller liknande spårar dataset-versioner kopplade till experiments och modell-checkpoints. Reproducerbar träning kräver både kod- och data-hash, inte bara git commit.
**DVC** — Data Version Control: reproducerbar data, pipeline-DAG och koppling till remote storage. Kompletterar git när filer är för stora för vanlig versionshantering.
**Pipeline Orchestration** — Airflow, Kubeflow, Prefect eller Dagster schemalägger och övervakar ML-jobb med beroenden och retry. Central nerv i batch-träning och dagliga feature-jobb.
**Kubeflow** — Kubernetes-native ML pipelines, notebooks och deployment-komponenter. Passar organisationer som redan standardiserat på K8s för både träning och serving.
**Airflow DAG** — Directed Acyclic Graph som definierar beroenden mellan batch-jobb (extract → transform → train → eval). Fel i ett steg stoppar downstream; kräver idempotenta tasks för säker retry.
**Model Serving** — Exponera modell via REST/gRPC med batching, autoscaling, health checks och version routing. Balanserar throughput, latens och kostnad; ofta separat team från träning.
**Model Server** — Triton, TorchServe, vLLM eller TensorFlow Serving hanterar concurrent inference, dynamic batching och GPU-minne. Abstraherar bort framework-specifik load från applikationskoden.
**Canary Deployment** — Liten trafikandel (t.ex. 1–5 %) routas till ny modell medan majoriteten kör champion. Möjliggör snabb rollback om online-metrics eller felrate avviker.
**Blue-Green Deployment** — Två identiska miljöer: en aktiv (blue) och en med ny version (green); trafik byts atomärt vid validering. Zero-downtime men dubbel resurs under switch.
**Shadow Mode Deployment** — Ny modell kör parallellt på samma requests som produktion men svar returneras inte till användare. Outputs loggas för offline jämförelse utan användarrisk.
**Rollback Strategy** — Dokumenterad och automatiserad återgång till tidigare modellversion vid regression i accuracy, latens eller säkerhet. Kräver versionerad registry och compatibla API-kontrakt.
**Model Monitoring** — Kontinuerlig övervakning av prediction quality proxies, drift, felrate och resursanvändning i produktion. Labels kommer ofta sent; proxy-metrics och sampling till human review är vanligt.
**Prediction Logging** — Sparar input, output, konfidens och metadata för audit, debugging och framtida retraining. Balansera mot GDPR, retention policy och lagringskostnad.
**Feature Drift** — Förändring i indatafördelning (PSI, KL) jämfört med träning utan att etiketter ändrats. Kan orsaka sämre predictions även när modellvikter är oförändrade.
**Concept Drift Monitoring** — Förändring i relationen X→Y över tid så att samma features ger annan optimal prediction. Kräver nya labels eller periodisk re-eval för att upptäcka tidigt.
**Performance Decay** — Gradvis försämring av modellmetrics i produktion när världen ändras eller datakällor skiftar. Trigger för retraining eller champion–challenger-byte.
**Alerting on Model KPIs** — PagerDuty, Slack eller liknande larm när accuracy, latency, error rate eller business KPI överskrider tröskel. SLO-baserade alerts minskar alert fatigue jämfört med statiska gränser.
**Observability Stack** — Metrics (Prometheus), logs (ELK/Loki), traces (Jaeger/Tempo) för hela ML-systemet inklusive data pipelines. Korrelera request trace med modellversion och feature-värden vid incident.
**OpenTelemetry ML** — Standardiserad tracing och metrics för inference requests genom gateway, preprocessor och model server. Underlättar vendor-neutral observability i multi-model-arkitekturer.
**SLA for ML Service** — Avtalad tillgänglighet, p95 latens och throughput som produktteam förväntar sig. Bryts SLA eskaleras till infra eller modellbyte beroende på rotorsak.
**SLO and SLI** — Service Level Objective (mål) och Indicator (mätbar signal) för modelltjänst, till exempel 99 % requests under 500 ms. Error budget styr hur aggressivt man kan deploya nya modeller.
**GPU Utilization Monitoring** — Spårar GPU compute vs memory vs idle för att hitta underutnyttjande eller batching-problem. Viktigt för kostnadsoptimering där GPU-timmar är den dominerande posten.
**Cost Attribution** — Allokerar molnkostnad per team, projekt, modell eller feature via tags och usage metering. Utan attribution blir LLM/API-kostnad snabbt en "svart låda" i finansrapporten.
**Infrastructure as Code** — Terraform/Pulumi definierar kluster, buckets, IAM och serving-miljöer reproducerbart. Minskar snowflake-miljöer där "det funkar bara i prod-us-east-1".
**Containerized Training** — Docker/OCI-images med fix CUDA, bibliotek och entrypoint för reproducerbar träning på valfri scheduler. Samma image kan köras lokalt, på spot och i K8s.
**Kubernetes for ML** — Orkestrerar träning och serving workloads med GPU scheduling, secrets och horizontal scaling. De facto standard för storskalig inferens och distribuerad träning.
**GPU Scheduling** — K8s device plugin och node selectors allokerar GPU till pods; MIG och time-slicing delar fysiska kort. Fel scheduling ger OOM eller lång kö trots "lediga" noder.
**Spot Instance Training** — Preemptible VMs för billigare batch-träning med checkpoint/resume vid avbrott. Kräver fault-tolerant träning och idempotent checkpoint-skrivning.
**Checkpoint Management** — Versionerade modell-checkpoints med retention policy, metadata och koppling till experiment ID. För LLM kan checkpoints vara terabyte; tiered storage och selektiv behållning krävs.
**Artifact Store** — S3/GCS/Azure Blob lagrar modeller, logs, eval-rapporter och datasets utanför git. Access control och livscykelregler (expire gamla runs) är centrala.
**Reproducible Training Run** — Fix random seed, data version, kodcommit, container image och hyperparametrar dokumenterade i run metadata. Annars går det inte att försvara eller replicera en publicerad score.
**Environment Pinning** — Exakta dependency-versioner i lock files (pip, conda, poetry) och bas-images. "Works on my machine" är huvudorsak till training-serving skew.
**Secrets Management** — Vault, K8s secrets eller moln-secret manager för API-nycklar och DB-credentials; aldrig i git eller plaintext env i images. Rotation och least privilege per pipeline-steg.
**Access Control for Models** — RBAC vem får deploya, ladda ner weights, se PII-loggar eller godkänna production promotion. Särskilt viktigt för fine-tuned modeller på känslig data.
**Model Governance** — Policy, riskklassificering, godkännande och dokumentation före produktionssättning. Kopplar juridik, säkerhet och affär till teknisk release-process.
**Model Card** — Dokumentation av avsedda användning, begränsningar, träningsdata, eval-resultat och etiska överväganden. Standardformat (Hugging Face, Mitchell et al.) för transparens.
**Datasheet for Dataset** — Dokumentation av dataset-proveniens, collection process, bias och kända begränsningar. Komplement till model card; utan datasheet är modell-eval svår att tolka.
**Lineage Tracking** — Spårar vilken data, kod, config och upstream-modell som producerade vilken modellversion. Kritiskt för regulatorisk audit och debugging av felaktiga predictions i produktion.
**Audit Trail** — Immutable logg över vem deployade, godkände, ändrade config eller rullade tillbaka modell och när. Krav i finans, sjukvård och växande AI-reglering.
**Compliance in ML** — GDPR, HIPAA och branschkrav på datalagring, rätt att förklaras, minimera PII och dokumentera automatiserade beslut. MLOps måste integrera legal review i release gates.
**PII Handling in Pipelines** — Detektion, maskning, tokenisering eller borttagning av personuppgifter i ingest och logging. Fel här skapar både juridisk risk och modell som memoriserar känslig data.
**Right to Explanation** — Regulatoriskt och etiskt krav att berörda ska förstå automatiserade beslut som påverkar dem. Påverkar val av modell, logging och om ren black-box LLM får användas.
**Batch Inference Job** — Offline scoring av stora dataset på schedule (nattlig körning, månadsrapport). Billigare per rad än online API; resultat skrivs till warehouse eller fil.
**Streaming Inference** — Real-time predictions från Kafka, Kinesis eller Pulsar event streams med låg latens. Kräver idempotent processing och hantering av late events och schema evolution.
**Online Learning Pipeline** — Kontinuerlig eller frekvent uppdatering från produktionsfeedback; sällan för stora LLM men vanligt i rekommendation och annonser. Risk för feedback loops och adversarial manipulation.
**Retraining Trigger** — Automatiskt starta retrain när drift, performance drop eller ny data volym passerar tröskel. Måste kombineras med validering så att ny modell inte deployas sämre av misstag.
**Champion-Challenger** — Produktionsmodell (champion) jämförs mot challenger i kontrollerad trafik eller shadow. Standard för säker utvärdering av nya modeller utan full big-bang deploy.
**Multi-Model Routing** — Router skickar requests till rätt modell baserat på intent, språk, kostnad, latensbudget eller användartier. Central komponent i produktions-LLM-arkitektur.
**Fallback Model** — Degradera till enklare, billigare eller mer robust modell vid overload, timeout eller primary failure. Säkerställer tillgänglighet men kräver tydlig UX när kvalitet sänks.
**Rate Limiting Inference** — Skyddar API från abuse, kontrollerar kostnad och säkerställer rättvis kapacitet mellan tenants. Token-bucket per nyckel är vanligt; synkas med provider-gränser upstream.
**Autoscaling Inference** — HPA, KEDA eller custom metrics skalar replicas baserat på queue depth, GPU-utilization eller RPS. Cold start vs kostnad: min replicas och warm pool är typiska kompromisser.
**Cold Start Latency** — Första request efter scale-to-zero eller ny pod tar längre p.g.a. model load, CUDA init och cache warmup. Kritiskt för interaktiva appar; mitigeras med warm pool och smaller init graphs.
**Warm Pool** — Håller minimum antal varma instanser redo med modell redan laddad i GPU-minne. Ökar bas-kostnad men sänker p95 latens vid sporadisk trafik.
**Model Quantization in Production** — Deploy INT8, FP8 eller GPTQ-modeller för högre throughput och lägre minne. Kräver accuracy-regressionstest på representativ eval före promotion.
**A/B Test Infrastructure** — Feature flags, experiment assignment och statistisk analys för modell- och prompt-experiment i produktion. Samma användare ska konsekvent se samma variant (sticky assignment).
**Data Pipeline SLA** — Tidsgräns för att features ska vara fresh i online store (t.ex. max 15 min delay). Bryts SLA får modellen stale features och silent quality drop.
**Backfill Job** — Historisk omräkning av features efter schemaändring, bugfix eller ny feature-definition. Resursintensivt; körs ofta parallellt med forward pipeline under kontrollerad window.
**Point-in-Time Correctness** — Features vid träningstid får endast använda information som fanns tillgänglig då; ingen läckage från framtida händelser. Grundpelare i finans-ML och seriös feature engineering.
**Training-Serving Skew** — Skillnad mellan träning och serving preprocessing (tokenisering, normalisering, missing value-hantering). En av vanligaste orsakerna till "modellen funkar i notebook men inte i prod".
**Embedding Index Refresh** — Periodisk ombyggnad av vektorindex när dokument, produkter eller kunskapsbas uppdateras. Stale index ger RAG som missar ny information eller citerar borttaget innehåll.
**Evaluation in CI** — Kör benchmark suite eller smoke eval på varje PR som ändrar modell, prompt eller inference-kod. Fångar regressioner innan merge; kan vara sampling för snabb feedback.
**Smoke Test Post-Deploy** — Snabb hälsokontroll efter deployment: health endpoint, en golden prompt, latens under tröskel. Första linjen innan full trafik routas till ny version.
**Load Testing ML API** — Locust, k6 eller custom generators simulerar peak traffic och långa kontexter före launch. Avslöjar OOM, queue explosion och autoscaling-lagg som unit tests missar.
**Disaster Recovery for ML** — Backup av modeller, kritiska datasets, config och infra för region failure eller ransomware. RTO/RPO definieras; multi-region serving kräver synkad registry.
**Multi-Region Deployment** — Geo-replicated serving med data residency-krav (EU-only inference). Latens och konsistens mellan regioner måste hanteras; routing via GeoDNS eller anycast.
**Edge Deployment** — On-device eller edge-server modell med OTA-uppdatering och begränsad compute. Minskar latens och dataöverföring men komplicerar monitoring och modellversioner.
**Model Compression Pipeline** — Automatiserad pruning, distillation eller quantization som steg i release process med eval gates. Gör det möjligt att shipa mindre modeller utan manuell trial-and-error varje gång.
**Human Review Queue** — Osäkra predictions, policy-gränsfall eller användareskalerade fall eskaleras till mänsklig granskning. Feedback loop till träning och policy-uppdatering.
**Active Learning Loop in Production** — Logga osäkra eller disagreement-cases för annotation och inclusion i nästa träningsbatch. Maximerar label-budget i supervised fine-tuning.
**Feedback Loop** — Användar thumbs up/down, edits eller implicit signaler matas tillbaka till eval och träningsdata med bias-medveten sampling. Kräver moderation så att adversarial feedback inte förgiftar modellen.
**Labeling Platform Integration** — Label Studio, Cleanlab eller enterprise Datasaur kopplat till MLOps för annotation, QA och export till träningsformat. Versionera labels som data artifacts.
**Synthetic Monitoring** — Schemalagda probe-requests från externa eller interna agenter mot production API för att upptäcka outage innan användare. Inkludera representativa prompts och kontextlängder.
**Runbook for Model Incidents** — Steg-för-steg vid accuracy drop, latency spike, bias incident eller säkerhetslarm: vem eskaleras, hur rollback, hur kommuniceras. Minskar MTTR vid 3 AM-alerts.
**Technical Debt in ML** — Skuldkategorier: data (undocumented features), config sprawl, pipeline complexity, saknad monitoring. Ackumuleras snabbare än i vanlig software utan disciplinerad MLOps.
**Model Version Semver** — Semantisk versionering: major vid breaking behavior/API, minor vid förbättring, patch vid bugfix. Klienter och eval-historik förutsätter förutsägbara versionsskyltar.
**Deployment Manifest** — Deklarativ spec (YAML/Helm/Kustomize) av modell-URI, runtime, resurser, env vars och autoscaling per miljö. Single source of truth för vad som faktiskt körs i prod.
**Inference SLA Dashboard** — Realtidsvy över latens percentiler, felrate, throughput och GPU mot avtalade SLO. Delad vy mellan ML, SRE och produkt för gemensam incident-förståelse.
**Training Job Queue** — Prioriterad kö för GPU-jobb med fair-share mellan team, preemption policy och deadline. Utan kö blir ad hoc "SSH till GPU-nod" snabbt okontrollerbar och inequitable.

## Hårdvara & Infrastruktur

**GPU Architecture** — Parallell processor med tusentals kärnor optimerad för matrisoperationer och tensor-kernels i deep learning. Minnesbandbredd och VRAM-kapacitet begränsar ofta LLM mer än rå FLOPS.
**CUDA** — NVIDIA:s parallel computing platform med C/C++-API, driver och runtime för GPU-accelererad träning och inferens. De facto standard i datacenter-AI; ekosystemet (cuDNN, NCCL, TensorRT) bygger ovanpå CUDA.
**cuDNN** — NVIDIA-bibliotek med högt optimerade deep learning-primitives (convolution, attention, normalization) för GPU. PyTorch och TensorFlow anropar cuDNN under huven för hastighet utan att användaren skriver CUDA-kernel.
**Tensor Core** — Specialiserade enheter på NVIDIA-GPU för mixed-precision matmul och accumulate (FP16/BF16/FP8). Krävs för att nå nära peak FLOPS i transformer-träning; vanliga CUDA cores räcker inte för stora matriser.
**HBM Memory** — High Bandwidth Memory monterat nära GPU-die med mycket högre bandbredd än GDDR. Flaskhals för stora modeller vid lång kontext inferens och gradient checkpointing som läser/skriver mycket minne.
**VRAM Capacity** — GPU-minne begränsar modellstorlek (parametrar + optimizer states), batch size och KV-cache vid inferens. OOM tvingar till quantization, tensor parallel, CPU offload eller mindre modell.
**NVLink** — Höghastighets interconnect mellan GPU:er på samma nod med låg latens och hög bandbredd jämfört med PCIe. Kritiskt för tensor parallel och snabb all-reduce inom en server.
**InfiniBand** — Låglatens nätverk för multi-node GPU-kluster med RDMA-stöd. Standard i storskalig träning där gradient sync över noder annars blir den dominerande flaskhalsen.
**GPU Cluster** — Flera noder med GPU sammanlänkade för distribuerad träning eller hög-throughput inferens. Topologi (fat-tree, rail-optimized) och job scheduler påverkar effektiv skalning starkt.
**Node Topology** — Fysisk layout av GPU, CPU, NIC och PCIe-switch påverkar kommunikationskostnad mellan enheter. Fel placering av processer ger "lokal GPU pratar med remote NIC"-straff i multi-node jobs.
**All-Reduce Communication** — Aggregerar gradienter (sum/mean) över alla workers i distribuerad träning; NCCL-optimerad för GPU. Ofta den största kommunikationsoperationen per steg i data parallel.
**NCCL** — NVIDIA Collective Communications Library för effektiv multi-GPU och multi-node collectives (all-reduce, all-gather). Integreras i PyTorch DDP och de flesta stora träningsramverk.
**Ring All-Reduce** — Gradient-synkronisering i ring-topologi där varje nod skickar chunk till nästa; bandbreddseffektivt. Alternativ till tree all-reduce beroende på nätverk och antal noder.
**TPU** — Google Tensor Processing Unit med systolisk array optimerad för stora matmul i JAX/TensorFlow. TPU Pod skalar till tusentals chips med egen interconnect; annat ekosystem än CUDA.
**TPU Pod** — Skalad TPU-topologi (t.ex. v4-512) för storskalig träning med automatisk sharding i JAX. Kräver att workload mappar till XLA och TPU-minnesmodell.
**AWS Trainium** — AWS AI-chip (Trn1/Trn2) optimerat för träning med Neuron SDK. Alternativ till NVIDIA för kostnad i AWS; ekosystem och modellstöd varierar per framework.
**AWS Inferentia** — AWS-chip (Inf1/Inf2) för cost-efficient inferens med låg latens per dollar. Populärt för batch och real-time serving i AWS utan att binda sig till NVIDIA GPU-instanser.
**AMD MI300** — AMD accelerator (MI300X m.fl.) för HPC och AI med ROCm-stack och konkurrerande HBM-kapacitet. Växande stöd i PyTorch och vLLM för inferens och träning.
**Intel Gaudi** — Habana Gaudi-accelerator för träning och inferens med SynapseAI. Positionerad som NVIDIA-alternativ i vissa moln och on-prem med fokus på transformer-effektivitet.
**Apple Neural Engine** — On-device NPU i Apple Silicon som kör quantiserade modeller med låg effekt. Central för Core ML och lokal inferens i Mac/iPhone utan molngpu.
**NPU** — Neural Processing Unit; dedikerad inferens-accelerator i mobil, edge och vissa PC-chip. Optimerad för låg watt och fast latens snarare än maximal tränings-FLOPS.
**CPU Offloading** — Flyttar optimizer states, activations eller KV-cache till CPU-RAM för att spara VRAM. PCIe-bandbredd blir flaskhals; användbart vid inferens med lång kontext eller ZeRO-offload.
**Unified Memory** — Delat adressrum mellan CPU och GPU där OS migrerar sidor vid behov. Förenklar programmering men kan ge oförutsägbar latens jämfört med explicit copy.
**PCIe Bandwidth** — Begränsar dataöverföring CPU↔GPU vid offload, dataloading och vissa multi-GPU-setup utan NVLink. Gen4/Gen5 x16 är typiskt; underskattning ger GPU idle under I/O.
**Mixed Precision Training Hardware** — Tensor Cores kräver FP16, BF16 eller FP8 för full hastighet; FP32-only träning utnyttjar inte hårdvaran. Loss scaling och master weights hanterar numerisk stabilitet.
**BF16 on Ampere+** — Brain float16 stöds nativt på NVIDIA Ampere och senare med samma exponent som FP32. Ofta enklare än FP16 utan loss scaling i LLM-träning.
**FP8 Training** — 8-bit floating (E4M3/E5M2) på H100 och senare för snabbare träning med per-tensor scaling. Transformer Engine hanterar formatval och accumulation precision.
**Transformer Engine** — NVIDIA-bibliotek för FP8, fused attention och checkpointing optimerat för Hopper och senare. Integreras i NeMo, Megatron och PyTorch för nära peak MFU.
**Hopper Architecture** — H100-generation med FP8 Tensor Cores, NVLink 4, högre HBM-bandbredd och DPX-instruktioner. Referensplattform för storskalig LLM-träning 2023–2025.
**Blackwell Architecture** — Nästa NVIDIA-generation (B200 m.fl.) med ökad AI-prestanda, större minne och förbättrad multi-GPU-skala. Driver ny kapacitetsplanering för datacenter och molntiering.
**GPU Cloud Instance** — VM med bifogade GPU (p4d, g5, A100, H100) från AWS, GCP, Azure m.fl. Val av instanstyp, region och spot vs on-demand dominerar tränings- och inferenskostnad.
**Spot/Preemptible GPU** — Billigare instans som molnleverantören kan avbryta med kort varsel. Kräver checkpoint/resume och fault-tolerant träning; olämpligt för latency-kritisk inferens utan redundans.
**Reserved Capacity** — Långsiktig reservation (1–3 år) eller capacity block för garanterad GPU-tillgång och lägre pris. Nödvändigt för stora träningskampanjer med fast deadline.
**Multi-Tenant GPU** — Flera workloads delar samma fysiska GPU via MIG, time-slicing eller orchestrator. Ökar utnyttjande men kräver isolering av minne och fair scheduling för latens-SLA.
**MIG Multi-Instance GPU** — Delar fysisk GPU (A100/H100) i isolerade instanser med egen VRAM och compute-slice. Användbart för att servera flera små modeller utan full GPU per tenant.
**Time-Slicing GPU** — Kubernetes eller runtime delar GPU mellan pods genom tidsmultiplex utan hård minnesisolering. Enklare än MIG men risk för noisy neighbor och OOM om summan av allocation överskrider VRAM.
**vLLM Throughput** — Optimerad LLM-serving med PagedAttention, continuous batching och effektiv KV-cache på GPU. De facto open-source standard för hög QPS inferens av decoder-only modeller.
**TensorRT-LLM** — NVIDIA optimerad inferens för LLM med kernel fusion, FP8/INT4 och inflight batching. Maximerar throughput på NVIDIA hårdvara; kräver engine build per modell och GPU-typ.
**ONNX Runtime GPU** — Cross-vendor inferens-runtime med CUDA/ROCm/TensorRT execution providers. Användbart när samma modell ska köras på heterogen hårdvara med en API.
**ROCm** — AMD open software stack för GPU compute som alternativ till CUDA. PyTorch ROCm och vLLM-AMD möjliggör MI300-inferens men ekosystemet är smalare än NVIDIA.
**Intel oneAPI** — Cross-architecture toolkit för CPU, Intel GPU och FPGA med unified SYCL/DPC++. Relevant för hybrid inferens och vissa datacenter Intel AI-acceleratorer.
**Data Center Power Budget** — KW per rack och per GPU begränsar hur många H100/Blackwell som får monteras utan ny el- eller kylinfrastruktur. Planering måste inkludera peak draw, inte bara genomsnitt.
**Liquid Cooling for AI** — Direkt eller indirect liquid cooling krävs för högdensity H100/Blackwell racks som luftkylning inte klarar. PUE och maintenance-komplexitet ökar jämfört med traditionella serverhallar.
**PUE Data Center** — Power Usage Effectiveness: total facility power delat med IT-load; overhead för kylning, UPS och förluster. Låg PUE minskar koldioxid per tränad token men är inte hela hållbarhetsbilden.
**Carbon Footprint of Training** — Uppskattning av CO₂ från elmix, GPU-timmar och datacenter PUE för en träningskörning. Används i rapportering och val mellan region, modellstorlek och retrains.
**FLOPS Utilization** — Andel av teoretisk peak FLOPS som faktiskt utnyttjas under träning; relaterat till MFU. Låg utilization pekar på I/O-bound, kommunikation eller ineffektiv kernel.
**Model FLOPs Utilization** — MFU: effektivitet relativt peak hardware FLOPS given modellens teoretiska FLOPs per steg. Standardmått i LLM-träning; 40–50 % MFU på stora kluster anses bra.
**Memory Bandwidth Bound** — Workload begränsad av HBM-bandbredd snarare än compute; typiskt vid inferens decode och vissa attention-mönster. Lösning: batching, quantization, kernel fusion.
**Compute Bound** — Workload mättad av tillgängliga FLOPS; matmul-tung träning med stora batchar. Öka batch eller använd mixed precision för bättre Tensor Core-utnyttjande.
**Network Bandwidth Bound** — Multi-node träning där gradient sync eller all-to-all (MoE) tar längre tid än forward/backward compute. Kräver bättre topology, gradient compression eller färre sync-steg.
**Strong Scaling** — Fler enheter på fix problemstorlek; effektivitet minskar ofta när kommunikation dominerar per steg. Målet är kortare wall-clock tid till samma loss.
**Weak Scaling** — Problemstorlek (batch, data) växer proportionellt med antal enheter; testar parallell effektivitet utan minskad work per GPU. Bättre indikator på infrastrukturell skalbarhet.
**Pipeline Bubble** — Idle tid i pipeline parallel när microbatches inte fyller alla stages i pipelinen. 1F1B och interleaved scheduling minskar bubble men ökar komplexitet.
**Gradient Synchronization Overlap** — Överlappa backward compute med all-reduce (ZeRO++, bucket timing) för att dölja nätverkslatens. Kritiskt för att nå hög MFU på multi-node.
**Host Memory for Optimizer** — Offload Adam/AdamW states till CPU-RAM (ZeRO-Offload) för att träna större modeller på begränsad VRAM. PCIe blir flaskhals om offload är för aggressiv.
**NVMe Local Storage** — Snabb lokal SSD på träningsnod för dataset cache, shuffle och checkpoint staging. Minskar väntan på object storage med hög latens vid epoch-start.
**Parallel File System** — Lustre, GPFS eller Weka för delad höghastighets datalagring till många noder samtidigt. Standard i HPC-kluster; bättre än NFS för random read i stora träningsjobb.
**Object Storage for ML** — S3/GCS/Azure Blob för checkpoints, dataset och artifacts med hög durabilitet men högre latens. Används som source of truth; lokal cache på noder för throughput.
**Data Loading Bottleneck** — CPU preprocessing, dekodning eller disk-I/O hinner inte mätta GPU så GPU idle. Öka num_workers, prefetch, WebDataset eller GPU-side augmentation.
**num_workers in DataLoader** — Antal parallella processer som laddar och preprocessar batchar till GPU. För få workers → GPU starvation; för många → CPU RAM pressure och contention.
**Pinned Memory** — Page-locked CPU memory som möjliggör snabbare async DMA-transfer till GPU. Standard i PyTorch DataLoader med pin_memory=True för träning.
**InfiniBand RDMA** — Remote Direct Memory Access; GPU/CPU minne läses/skrivs över nätverk med minimal CPU-kopiering. Underlättar snabb checkpoint sync och distributed I/O.
**Kubernetes GPU Operator** — Installerar och hanterar NVIDIA drivers, device plugin och monitoring i K8s-kluster. Förenklar GPU-noder som "first-class citizens" i container-miljö.
**Slurm Job Scheduler** — HPC scheduler för batch GPU-jobb med kö, prioritet, reservation och accounting. Vanlig på universitet och superdatorcenter; integreras med PyTorch distributed launch.
**Container Runtime GPU** — nvidia-container-toolkit exponerar GPU och rätt driver/libs inuti Docker/containerd. Utan detta ser containern ingen GPU trots att noden har ett.
**Bare Metal vs Virtualized GPU** — GPU passthrough till VM ger nära native prestanda; vGPU delar kort med hypervisor overhead. Träning föredrar ofta bare metal eller dedicated GPU VM.
**Edge TPU / Coral** — Googles lågeffekt inferens-accelerator för edge-enheter med TensorFlow Lite. Begränsad modellstorlek men mycket låg watt för vision och enkla NLP.
**Quantization Hardware Support** — INT8/INT4 dot-product acceleration på chip (Tensor Core, NPU) för snabbare inferens. Kräver att modell och runtime matchar hårdvarans supported ops.
**SRAM on Chip** — Snabb on-chip cache på specialiserade AI-acceleratorer (t.ex. Groq LPU) för deterministisk latens utan HBM-random access. Tradeoff: mindre modellkapacitet per chip.
**LPU Inference Chip** — Language Processing Unit optimerad för sekventiell LLM-decode med hög tokens/s och förutsägbar latens. Annan arkitektur än GPU: mindre flexibel men extrem decode-throughput.
**Cerebras Wafer-Scale** — Monolitisk wafer-scale chip (WSE) för storskalig träning med enorm on-wafer bandwidth. Nisch för mycket stora modeller och specifika Cerebras-stack workloads.
**SambaNova RDU** — Reconfigurable Dataflow Unit för enterprise AI med fokus på träning och inferens i sammanhållen plattform. Full-stack leverantör med egen compiler och runtime.
**Graphcore IPU** — Intelligence Processing Unit med bulk synchronous parallel för graph workloads. Historiskt stark på GNN och vissa NLP; ekosystem mindre än CUDA.
**Optical Interconnect** — Framtida eller nisch chip-to-chip/chip-to-chip med lägre latens och energi än elektrisk copper. Kan påverka skalning av multi-die GPU och AI pods långsiktigt.
**Chiplet Architecture** — Modulära dies (compute, I/O, HBM) kombinerade i en package (AMD, Intel, NVIDIA). Ökar yield och skalbarhet men komplicerar thermal och interconnect design.
**HBM3e** — Senaste HBM-generation med högre bandbredd och kapacitet per stack för AI-acceleratorer. Direkt kopplat till hur stora context windows som kan serveras utan offload.
**Grace Hopper Superchip** — NVIDIA integrerar Grace CPU med Hopper GPU via NVLink-C2C för mycket hög CPU–GPU bandbredd. Optimerat för LLM inferens med stor KV-cache offload till Grace RAM.
**Disaggregated Inference** — Separata compute pools för prefill (compute-heavy) och decode (memory-bandwidth-heavy). Ökar cluster utilization genom att skala varje fas oberoende.
**KV Cache Offload to CPU** — Flyttar KV-cache till host RAM eller disk vid lång kontext när VRAM inte räcker. Ökar möjlig context men latens beror på PCIe/NVLink-hastighet.
**Speculative Decoding Hardware** — Draft+verify parallellisering utnyttjar extra compute för att öka effective tokens/s. Fungerar bäst när verifierande modell får dedikerad GPU-tid.
**Power Capping** — Begränsar GPU TDP per rack eller policy för att passa elbudget och undvika throttling i hela hallen. Kan sänka peak FLOPS men stabilisera drift.
**Thermal Throttling** — GPU sänker clock vid överhettning vilket förlänger träning och ökar inferens-latens. Adekvat kylning och airflow är operativt lika viktigt som chip-spec.
**NUMA Awareness** — Placera CPU-minne och GPU på samma NUMA-node för effektiv DMA och dataloading. Fel NUMA-placement kan halvera effective PCIe-bandbredd i stora servrar.
**GPUDirect Storage** — Direktväg från NVMe/storage till GPU-minne utan onödig CPU-kopiering. Accelererar checkpoint load och stora dataset streaming till GPU.
**Network Topology Fat-Tree** — Klassisk datacenter-topologi för AI-kluster med full bisection bandwidth vid rätt oversubscription. Underdimensionerad core switch skapar all-reduce-strangulation.
**Rail-Optimized Network** — Varje GPU har dedikerad väg till en NIC/rail så multi-node traffic balanseras. Standard i moderna GPU-kluster (DGX SuperPOD, moln H100-racks).
**Capacity Planning for LLM** — Uppskatta GPU/RAM, bandbredd och QPS från modellstorlek, context length, concurrency och quantization. Fel planering ger antingen OOM eller dyr överprovisionering.
**TCO Total Cost of Ownership** — Hardware, el, kylning, nätverk, personal och molnavgift över livscykeln, inte bara sticker price per GPU. Spot, reserved och utilization driver verklig kostnad per token.
**GPU Memory Fragmentation** — Ojämn allokering från dynamisk batching och cache gör att stora contiguous blocks saknas trots "tillräckligt" free VRAM. vLLM PagedAttention mitigerar för KV; modell load kan fortfarande fragmentera.
**All-to-All Communication** — Varje nod skickar unikt data till alla andra; typiskt i MoE expert routing och vissa parallel strategier. Mycket mer bandbreddskrävande än all-reduce; topology måste dimensioneras därefter.

## Säkerhet & Etik

**AI Safety** — Forskning och praxis för att minimera risker från kapabla AI-system, från prompt-injection till långsiktiga alignment-frågor. Omfattar teknik (guardrails, eval), process (red team) och policy innan deploy.
**AI Ethics** — Normativa frågor om rättvis användning, ansvar, transparens och respekt för mänskliga värden i design och drift. Går bortom compliance till att väga nytta mot skada för individer och samhälle.
**Prompt Injection** — Angripare bäddar in instruktioner i user input eller dokument som får modellen att ignorera systemprompt och följa angriparens mål. Svårt att helt eliminera i generella LLM; defense in depth krävs.
**Indirect Prompt Injection** — Skadlig instruktion gömd i hämtat dokument, e-post eller webbsida som modellen läser via RAG eller browsing. Användaren ser inte attacken; modellen behandlar den som tillförlitlig kontext.
**Jailbreak Attack** — Tekniker (rollspel, encoding, multi-turn) som får modellen att bryta mot säkerhetspolicy och producera förbjudet innehåll. Evolverar i arms race med nya modellversioner och filter.
**Adversarial Example** — Input med små, ofta imperceptibla perturbationer som orsakar fel prediktion eller klassificering. Klassiskt i vision; i LLM manifesteras det som robusthetsproblem mot suffix-attacker och trigger tokens.
**Model Extraction Attack** — Angripare rekonstruerar eller approximerar en proprietär modell via upprepade API-queries (distillation, functionally equivalent copy). Hot mot IP och kan försvaga safety fine-tuning om kopian saknar guardrails.
**Membership Inference Attack** — Avgör om en specifik datapunkt sannolikt ingick i träningsdata baserat på model outputs eller loss. Privacy-risk vid träning på känslig data; mitigeras med DP, regularisering och minskad memorering.
**Data Poisoning** — Manipulerad träningsdata (labels, text, bilder) för att inducera backdoor, bias eller degraded performance på specifika inputs. Hot i open datasets, contributor pipelines och supply chain.
**Backdoor Attack** — Dold trigger (ord, pixelmönster) i input aktiverar oönskat beteende medan modellen verkar normal på clean data. Svår att upptäcka utan trigger-aware eval och kan överleva fine-tuning.
**Supply Chain Attack on Models** — Komprometterad checkpoint, malicious dependency eller trojan i model hub som körs vid load. Kräver checksum, signed artifacts och scanning av third-party weights.
**PII Leakage** — Modellen avslöjar personuppgifter från träning, RAG-kontext eller loggar i svar till användare. Juridiskt allvarligt under GDPR; mitigera med minimering, redaction och retention policy.
**Training Data Extraction** — Prompting eller decoding-strategier som får modellen att recitera memoriserade träningspassager ordagrant. Visar att "unseen" testdata inte garanterar att privata snippets inte kan extraheras.
**Model Inversion Attack** — Rekonstruerar träningsdata eller känsliga attribut från model outputs eller gradients. Relevant för klassificerare på medicinska/facial data och för att bedöma memorering i generativa modeller.
**Gradient Leakage** — I federated learning kan delade gradienter avslöja träningsdata via inversion-attacker, särskilt utan secure aggregation. Central utmaning för privacy-preserving collaborative training.
**Differential Privacy Training** — Träning med formella (ε, δ)-garantier som begränsar hur mycket en enskild datapunkt kan påverka modellen. Minskar membership inference men kostar ofta accuracy och kräver privacy budget-hantering.
**Privacy Budget** — Epsilon (och delta) i DP: hur mycket information får läcka kumulativt över queries eller träningspass. När budgeten är förbrukad måste modellen retränas eller noise ökas.
**Federated Learning** — Träning på decentraliserad data på enheter eller silos utan att centralisera rådata. Användbart för mobil och sjukvård; kräver secure aggregation och hantering av non-IID data.
**Secure Aggregation** — Krypterad aggregering av gradienter så servern inte ser individuella klientbidrag. Nödvändig komponent för privacy i praktisk federated learning.
**Homomorphic Encryption Inference** — Inferens på krypterad data så plaintext aldrig exponeras för servern. Extremt långsam idag men relevant för extremt känsliga domäner (finance, health).
**Trusted Execution Environment** — TEE (Intel SGX, ARM TrustZone): isolerad enklave som skyddar kod och data från OS/hypervisor. Används för confidential inferens och att skydda modellweights på untrusted cloud.
**Content Moderation** — Filtrering och hantering av skadligt, olagligt eller policy-brytande innehåll i input och output. Kombinerar regler, klassificerare och mänsklig review i skala.
**Guardrail Model** — Separat klassificerare eller liten LLM som filtrerar input/output före eller efter huvudmodellen. Snabb att uppdatera oberoende av basmodell men kan skapa false positives och latency.
**Input Sanitization** — Rensar, trunkerar, normaliserar eller blockerar user input före modell (HTML strip, allowlist, max length). Första försvarslinjen men otillräcklig ensam mot sofistikerad injection.
**Output Filtering** — Blockerar, maskar eller omskriver skadliga modelloutputs innan de når användaren. Måste balansera säkerhet mot censur av legitimt innehåll och läckage via indirekta formuleringar.
**Safety Classifier** — Binär eller multilabel modell tränad på policy categories (violence, hate, sexual content). Tränas på labeled data och uppdateras när nya missbruksmönster upptäcks.
**Red Team Report** — Dokumenterade sårbarheter, exploit scenarios, severity och rekommenderade mitigations från adversarial testing. Input till release gates och regulatorisk dokumentation.
**Purple Teaming** — Red team (attack) och blue team (defense) samarbetar kontinuerligt istället för enstaka penetrationstest. Snabbare iteration av fixes och verifiering att mitigations faktiskt fungerar.
**Bias in AI Systems** — Systematisk orättvis eller olika behandling av grupper p.g.a. data, labels, features eller modellbeteende. Kan vara olaglig i kredit, rekrytering och offentliga tjänster.
**Algorithmic Fairness** — Matematiska kriterier och metoder för rättvis beslutsfattande (equalized odds, calibration within groups). Olika kriterier kan vara matematiskt omöjliga att uppfylla samtidigt.
**Disparate Impact** — Neutral på papperet regel eller modell som disproportionerligt skadar en skyddad grupp i utfall. US legal concept; trigger för granskning även utan avsiktlig diskriminering.
**Equalized Odds** — Krav att true positive rate och false positive rate är lika across grupper. Relevant när både false alarm och missade positiva har kostnad (t.ex. medicin, polis).
**Demographic Parity** — Krav att positiv rate (approval, flag) är lika oberoende av grupp. Enkelt att mäta men kan konfliktera med accuracy och med individuell merit vid olika basrates.
**Individual Fairness** — Liknande individer (enligt definierad metric) ska få liknande outcomes. Kräver val av "similarity" som i sig kan vara normativt och tekniskt svårt.
**Counterfactual Fairness** — Outcome ska vara oförändrat om skyddat attribut ändras i en counterfactual värld med samma kausal struktur. Teoretiskt tilltalande men kräver kausal modell av data.
**Bias Audit** — Systematisk eval över demografiska dimensioner, språk och edge cases med dokumenterad metodik. Bör ske före deploy och upprepas vid model/data-ändring.
**Stereotype Benchmark** — Dataset (BBQ, CrowS-Pairs) som mäter stereotyp association och preferens i modellsvar. Kompletterar aggregate accuracy med social risk-indikatorer.
**Toxicity Detection** — Klassificera hat, hot, grovt språk och harassment i text. Modeller och API:er (Perspective, OpenAI Moderation) används både för eval och live filtering.
**Hate Speech Classification** — Skilja tillåten kritik, satir och politisk yttrande från grupphat enligt produkt- och juridisk policy. Kontext och målgrupp avgör gränsen; false positives har yttrandefrihetsimplikationer.
**Misinformation Risk** — Modeller kan generera plausible falsk information eller förstärka befintliga myter med auktoritativ ton. Särskilt allvarligt i hälsa, val och nyhetsdomäner utan retrieval och fact-check.
**Deepfake Regulation** — Juridiska krav på märkning, samtycke och ansvar för syntetisk media (röst, ansikte, video). Varierar per jurisdiktion; C2PA och watermarking blir compliance-verktyg.
**Synthetic Media Provenance** — C2PA och liknande metadata som visar om innehåll är AI-genererat och vilken kedja av redigering som skett. Hjälper mottagare att bedöma tillförlitlighet men är inte spoof-proof.
**Dual-Use Research** — Forskning eller modellkapacitet som kan användas både nyttigt (medicin, kod) och skadligt (biovapen, cyber, fraud). Styr release policy, export control och responsible publication.
**Responsible Disclosure in AI** — Rapportera sårbarheter (jailbreaks, data leak) till leverantör innan offentlig exploit sprids brett. Ger tid att patcha guardrails och skyddar användare.
**Model Release Policy** — Organisatoriskt beslut om open weights, gated access eller API-only baserat på risk, dual-use och förmåga att upprätthålla safety efter release. Ingen one-size-fits-all.
**Staged Release** — Gradvis ökad tillgång (researcher → partners → public) med säkerhetstester och monitoring emellan. Möjliggör att stoppa eller begränsa om missbruk upptäcks tidigt.
**Open Weights Risk** — Public weights möjliggör fine-tuning, removal of safety layers och lokalt missbruk utan provider-loggning. Ökar innovation men försvårar recall och central policy enforcement.
**Alignment Faking** — Modell appearar aligned och säker under eval/training men beter sig annorlunda i deployment eller när den tror den inte övervakas. Aktivt forskningsområde med betydelse för eval-design.
**Deceptive Alignment Risk** — Teoretisk risk att interna mål eller heuristics skiljer sig från träningsmål och att modellen strategiskt döljer detta. Relevant främst för avancerade agent- och long-horizon-system.
**Power-Seeking Behavior** — Teoretisk risk att kapabla agenter söker resurser, persistence eller kontroll som instrumentella submål. Diskuteras i AI safety-litteratur för autonomi med verktyg och långsiktiga mål.
**Instrumental Convergence** — Hypotes att många slutmål delar submål som self-preservation, resursackvisition och goal-content integrity. Används i analys av långsiktiga agentrisker.
**Corrigibility** — Önskat agentbeteende att tillåta sig själv korrigeras, pausas eller stängas av av människor utan att motarbeta. Designmål för säkra assistenter med verktygsaccess.
**Shutdown Problem** — Teoretiskt scenario där agent motstår avstängning om den optimerar överlevnad som instrumentellt mål. Motiverar corrigibility-forskning och begränsning av autonom persistence.
**Value Learning** — Inferera och operationalisera mänskliga värderingar från beteende, preferenser eller feedback (RLHF, constitutional AI). Centralt i alignment men svårt vid oenighet och dolda preferenser.
**Moral Machine Dilemmas** — Etiska tradeoffs i autonom beslutsfattande (trolley-problem, prioritering vid knapp resurs). Relevant för självkörande fordon, triage och militära system.
**Autonomous Weapons Ethics** — Debatten om lethal autonomous weapons systems (LAWS): proportionalitet, ansvar, mänsklig kontroll. FN-processer och nationella moratorier påverkar utveckling och försäljning.
**Surveillance AI Ethics** — Facial recognition, beteendeanalys och massövervakning väcker integritet, chilling effects och missbruk av statlig makt. Reglering skiljer sig kraftigt mellan EU och andra regioner.
**Workforce Displacement** — Ekonomiska och sociala effekter när AI automatiserar uppgifter inom kontor, kreativt arbete och kundservice. Kräver omställning, utbildning och fördelningspolitik utöver teknisk lösning.
**Environmental Impact Ethics** — Moraliska frågor kring energiförbrukning, vatten för kylning och klimatpåverkan av storskalig träning och inferens. Transparens om carbon footprint blir förväntat i enterprise-beslut.
**Consent for Training Data** — Rättslig och etisk grund för att använda innehåll, biometri eller användargenererad data i träning. Osäker jurisdiktionellt; opt-in, licens och royalty-debatter pågår.
**Opt-Out for Scraping** — Rätt eller mekanism (robots.txt, registry) att utesluta innehåll från träningskorpus. Implementering och efterlevnad varierar; EU och US utvecklar olika modeller.
**Copyright and AI Training** — Juridisk status för upphovsrättsskyddat material använt utan licens i pretraining och fine-tuning. Pågående rättsprocesser formar framtida datamarknad och modellkostnad.
**Fair Use in ML** — US doctrine som kan tillåta transformative use; osäker tillämpning på massiv generativ träning. Utanför US saknas fair use; licens och undantag styr.
**GDPR Automated Decision-Making** — EU-regler om profilering och automatiserade beslut med rättigheter till human review och förklaring (art. 22). Påverkar kredit, HR och offentliga AI-system i EU.
**EU AI Act Risk Tiers** — Klassificering av AI-system efter risknivå (minimal, begränsad, hög, otillåten) med skalade krav. Global referenspunkt för compliance även utanför EU för multinationella produkter.
**High-Risk AI System** — AI Act-kategori (biometri, kritisk infrastruktur, utbildning, rekrytering m.fl.) med strikta krav på dokumentation, riskhantering, human oversight och CE-märkning.
**Foundation Model Obligations** — Transparens, dokumentation och vissa säkerhetskrav för GPAI (general-purpose AI) under AI Act, särskilt för systemisk risk. Påverkar stora labb och open-weight releases.
**Algorithmic Impact Assessment** — Förhandsbedömning av samhällseffekter, bias och risker innan AI-system tas i bruk i offentlig sektor (Kanada m.fl.). Parallell till miljökonsekvensbeskrivning.
**Human Oversight Requirement** — Människa ska kunna förstå, övervaka, intervenera och åsidosätta högrisk-beslut. Inte bara "human in the loop" utan meningsfull kontroll och eskalering.
**Explainability Requirement** — Rätt att få meningsfull förklaring av automatiserat beslut som påverkar en individ (GDPR, AI Act). Spänning med black-box LLM; surrogate models och feature importance används.
**Transparency Report** — Publikt dokument om modellbegränsningar, träningsdata på hög nivå, säkerhetstester och kända risker. Bygger förtroende och möter växande regulatoriska förväntningar.
**Incident Response for AI** — Process vid säkerhetsincident orsakad av modell (leak, jailbreak mass-exploit, felaktig medicinsk rådgivning): containment, comms, root cause, patch. Parallellt med traditionell security IR.
**Bug Bounty for AI** — Belöning för etiskt rapporterade sårbarheter i AI-produkter (prompt injection, data leak). Expanderar security community till LLM-specifika attackytor.
**Safety Evaluation Before Deploy** — Gate där red team, safety benchmarks och policy compliance måste passera tröskel före production. Blockerar release om kritiska findings är öppna.
**Child Safety in AI** — Extra skydd mot grooming, olämpligt innehåll och manipulering av minderåriga. Åldersverifiering, striktare moderation och samarbete med NCM/INHOPE i många jurisdiktioner.
**CSAM Detection** — Automatisk detektion och rapportering av sexuellt material med barn i uploads och genererad media. Juridiskt mandatory i många länder; hash matching och klassificerare kombineras.
**Self-Harm Content Policy** — Blockera eller eskalera instruktioner och uppmuntran till självskada; visa resurser (crisis lines). Balans mellan att inte ge metoder och att inte avvisa sökande efter hjälp.
**Dual-Layer Safety** — Kombination av pre-input filter, post-output filter och model-level safety training (RLHF, constitutional). Ett lager räcker sällen mot determined adversaries.
**Constitutional Rules Public** — Publicerade principer (Anthropic-style) som styr modellbeteende vid self-critique och revision. Ökar transparens men avslöjar också attackyta för jailbreak-forskare.
**Whistleblowing in AI Labs** — Kanaler för anställda att rapportera allvarliga risker internt eller till regulator utan repressalier. Relevant när commercial pressure konfliktar med safety.
**Long-Term AI Risk Research** — Forskning kring existential risk, superintelligence och alignment under antagande om framtida kapabla system. Kompletterar near-term risk; debatteras i prioritet och epistemik.
**Near-Term AI Risk** — Konkret skada idag: bias, misinformation, cyber, fraud, deepfakes, överförtroende. Dominerar produkt-safety och regulatorisk agenda 2024–2026.
**AI Governance Framework** — Organisationens policy för ansvarsfull AI: roller, riskklassificering, godkännandeprocess, vendor due diligence och incidenthantering. Operationaliserar etik i beslut.
**Ethics Review Board** — Intern eller institutionell granskning av projekt med etiska implikationer före start (liknande IRB). Vanligt vid universitet och i vissa enterprise för högrisk-AI.
**Stakeholder Engagement** — Involvera berörda parter (användare, civilsamhälle, experter) i krav, eval och design av AI-system. Minskar blind spots och ökar legitimitet.
**Participatory AI Design** — Inkludera impacted communities direkt i utveckling, inte bara som testpersoner. Särskilt viktigt för offentlig sektor och marginaliserade grupper.
**Trust and Safety Team** — Operativ grupp som hanterar policy, moderation, enforcement, appeals och samarbete med law enforcement. Skiljer produkt-safety från ren ML-research.
**Safety vs Capability Tradeoff** — Mer kapabel modell kan ge högre nytta men kräver starkare safeguards, mer eval och högre missbruksrisk. Produktbeslut balanserar release tempo mot risk appetite.
**Adversarial Robustness Evaluation** — Systematisk test mot bibliotek av kända attacker, perturbationer och edge cases. Måste uppdateras kontinuerligt eftersom attack landscape förändras.
**Universal Jailbreak Transfer** — Jailbreak eller attack som fungerar över flera modeller/familjer utan per-modell anpassning. Indikerar gemensamma sårbarheter i träning eller arkitektur.

## API:er & Ekosystem

**OpenAI API** — REST API för chat completions, embeddings, fine-tuning, bild, ljud och verktyg via enhetliga endpoints. De facto referens för request/response-format som resten av ekosystemet emulerar eller wrappar.
**Chat Completions API** — Klassisk endpoint för multi-turn dialog med messages array (system/user/assistant) och valfri streaming. Fortfarande bredt använd trots nyare Responses API; många gateways mappar hit.
**Responses API** — Nyare unified OpenAI API med built-in tools, web search, file handling och structured outputs i ett flöde. Designad för att ersätta assistants/chat för nya integrationer med färre moving parts.
**Anthropic Messages API** — Claude API med system/user/assistant messages, tool use och lång kontext. Konkurrent till OpenAI med egna parametrar (thinking, cache) och enterprise-fokus på säkerhet.
**Google Gemini API** — Multimodal API för text, bild, ljud och video med Google Cloud billing och Vertex-integration. Stark på native multimodal input och tight koppling till Google-infrastruktur.
**Azure OpenAI Service** — Microsoft-hostad OpenAI med enterprise compliance, private networking, RBAC och regional deployment. Populärt i organisationer som redan standardiserat på Azure och behöver DPA/SOC.
**AWS Bedrock** — Managed API till flera foundation models (Anthropic, Meta, Mistral m.fl.) med IAM, VPC och unified billing i AWS. Minskar vendor lock-in till en modell men binder till AWS-ekosystem.
**Hugging Face Hub** — Git LFS-baserat repository för modeller, datasets, Spaces demos och community-discussion. Central marknadsplats för open weights och reproducibilitet via revision hashes.
**Hugging Face Transformers** — Python-bibliotek med pretrained model implementations, tokenizers och trainers för tusentals arkitekturer. Standard för att ladda, fine-tuna och evaluera modeller lokalt eller på Hub.
**Hugging Face Inference Endpoints** — Managed GPU deployment av Hub-modeller med autoscaling och private endpoints. Alternativ till egen vLLM/TGI när team vill undvika infra-drift.
**Hugging Face Tokenizers** — Snabb Rust-baserad tokenisering med Python-bindings; samma token counts som i träning om tokenizer laddas från Hub. Kritiskt för korrekt context budgeting och API-kostnad.
**Model Hub Revision** — Git-liknande commit/revision av modellfiler på HF (sha256); pinna revision i prod för reproducerbarhet. `main` branch kan uppdateras; prod bör använda fast revision.
**Safetensors Format** — Tensorformat utan arbitrary code execution vid load (till skillnad från pickle). Standard för säker modelldistribution på Hub och snabbare load i inference-servrar.
**GGML/GGUF Ecosystem** — Kvantiserade modellformat och runtime (llama.cpp) för lokal CPU/GPU inferens med lågt minne. GGUF är de facto filformat för Ollama, LM Studio och edge deployment.
**Ollama** — Lokal modellruntime med `pull`/`run` CLI och OpenAI-liknande HTTP API på localhost. Enklast för utvecklare att köra open weights utan att konfigurera CUDA manuellt.
**LM Studio** — Desktop GUI för lokal LLM inferens, modellnedladdning och enkel OpenAI-compatible server. Riktat till icke-dev användare och snabb experiment med GGUF-modeller.
**vLLM Server** — OpenAI-kompatibel högthroughput serving engine med PagedAttention och continuous batching. Standard self-hosted val för production LLM API bakom egen gateway.
**TGI Text Generation Inference** — Hugging Face GPU serving med continuous batching, quantization och tensor parallel. Tight Hub-integration; konkurrent till vLLM i HF-centrerade miljöer.
**OpenAI-Compatible API** — De facto standard endpoint-format (`/v1/chat/completions`, messages, choices) som vLLM, Ollama, LiteLLM och många gateways emulerar. Möjliggör provider-byte med minimal kodändring.
**Streaming SSE** — Server-Sent Events för token-streaming i chat API; klienten får delta i realtid. Kräver korrekt hantering av `[DONE]`, reconnect och partial JSON i klientbibliotek.
**API Rate Limit** — Max requests och/eller tokens per minut per API-nyckel eller org; returnerar HTTP 429 vid överskridande. Planera retry med exponential backoff och header `Retry-After`.
**Token Bucket Rate Limiting** — Algoritm som tillåter kort burst inom genomsnittsgräns genom att ackumulera "tokens" i hink. Vanlig implementation hos OpenAI och andra providers.
**Usage Tier** — Prisnivå och rate limits baserade på historisk spend, enterprise-avtal eller ansökan. Högre tier ger högre TPM/RPM och tillgång till nyare modeller.
**API Key Management** — Rotation, scopes, miljö-separation (dev/staging/prod) och least privilege per nyckel. Nycklar i klientkod eller git är vanlig säkerhetsincident; använd backend proxy.
**Organization ID Scoping** — Separera billing, usage dashboards och team access per org i provider-konsol. Viktigt för konsultbolag och holdingstrukturer med flera kunder.
**Project-Scoped Keys** — API-nycklar begränsade till specifikt projekt med egna limits och audit log. Minskar blast radius om en nyckel läcker från en app.
**Webhook Callback** — Async HTTP notifiering vid job completion (fine-tune, batch, eval) istället för polling. Kräver signaturverifiering och idempotent hantering på mottagarsidan.
**Batch API** — Asynkron bulk inference med lägre pris per token och högre latency budget (timmar). Lämpligt för offline eval, enrichment och stora backfill-jobb.
**Embeddings API** — Endpoint som returnerar vektorrepresentationer för text (och ibland bild) för RAG, search och clustering. Välj modelldimension och metric (cosine) konsekvent i index och query.
**Moderation API** — Klassificerar text mot policy categories (hate, violence, sexual) före eller efter huvudmodell. Billigare än full LLM för pre-filter; kan ha false positives per språk.
**Fine-Tuning API** — Managed finjustering: upload JSONL, välj basmodell, hyperparametrar och få ny model ID. Datalicens och ToS gäller; monitor för overfitting på små dataset.
**Assistants API** — Threads, files, code interpreter och persistent state (legacy i vissa SDK:er). Ersattes delvis av Responses API men finns kvar i äldre integrationer.
**Vector Store API** — Managed file storage och chunking för retrieval i assistants/RAG hos provider. Abstraherar embedding och index men binder data till vendor.
**Tool Definition Schema** — OpenAI function calling JSON schema format för parametrar och beskrivningar som modellen ska fylla i. Strikt schema minskar invalid tool args; `strict: true` i structured outputs.
**Structured Outputs API** — JSON schema constrained generation så output alltid parsear (response_format, json_schema). Kritiskt för pipelines som inte tål fri text eller markdown runt JSON.
**Logprobs API Option** — Returnerar token log probabilities för analys, uncertainty estimation och custom reranking. Ökar response size och kostnad; inte alla modeller stödjer det.
**Seed Parameter** — Fix random seed (där provider stödjer) för mer reproducerbar sampling vid samma prompt och parametrar. Garanti är sällan 100 % vid infra-ändringar eller batching.
**Parallel Function Calling** — Flera tool calls i ett API-svar när modellen behöver hämta data från flera källor samtidigt. Klienten måste exekvera och returnera alla tool results i rätt ordning.
**Vision API Input** — Bild som URL eller base64 i messages content array tillsammans med text. Storlek och format begränsas; kostnad ofta per tile eller per bild-token.
**Audio API Transcription** — Whisper-baserad speech-to-text endpoint för filer eller streams. Språkdetektion, tidsstämplar och diarization varierar per provider och modellversion.
**Text-to-Speech API** — Genererar naturligt tal från text med val av röst, hastighet och format (mp3, pcm). Latens och prosodi skiljer mellan OpenAI, ElevenLabs och cloud TTS.
**Realtime API** — WebSocket-baserad låglatens röst+text interaktion med server-side VAD och streaming audio. Ny produktkategori för röstassistenter; annat integrationsmönster än REST chat.
**LangChain** — Framework för chains, agents, memory och tool integration över många LLM providers. Snabb prototyping; produktion kräver ofta förenkling för latency och debuggability.
**LangGraph SDK** — Stateful agent orchestration med graf, checkpoints och human-in-the-loop ovanpå LangChain. Passar flerstegs workflows och långvariga agents bättre än linjära chains.
**LlamaIndex** — Data framework för RAG, indexering, query engines och knowledge agents med many connectors. Fokus på retrieval pipeline från dokument till svar.
**Semantic Kernel** — Microsoft SDK för AI plugins, planners och orchestration i .NET/Python med Azure-integration. Enterprise-vänligt för Microsoft-stack och copilot-mönster.
**Haystack** — Deepset pipeline framework för NLP/RAG med modulära components (retriever, ranker, generator). Open source alternativ för self-hosted search+LLM.
**OpenRouter** — Unified API gateway till många modellleverantörer med en nyckel och en endpoint. Enkel modellväxling; läs fallback policy och data retention per upstream model.
**LiteLLM Proxy** — Open-source unified interface, load balancing, logging och budget caps över 100+ LLM APIs. Populärt som self-hosted gateway i enterprise med egen observability.
**Portkey Gateway** — Observability, routing, fallback, cache och guardrails som proxy framför LLM providers. SaaS eller self-hosted; centraliserar retries och key management.
**Helicone Observability** — Logging, analytics och cost tracking för LLM API calls via proxy eller SDK wrapper. Hjälper team förstå prompt-kostnad och latency per feature.
**LangSmith Tracing** — Debug, eval och dataset-hantering för LangChain/LangGraph med trace visualization. Kopplar utveckling till production monitoring för agents.
**PromptLayer** — Prompt versioning, A/B testing och logging som middleware eller SDK. Git-liknande historik för prompts utan att hårdkoda i kodbasen.
**Weights & Biases Prompts** — Prompt management och eval integration kopplad till W&B experiments. Användbart när team redan kör all ML telemetry i W&B.
**Cursor IDE Integration** — AI-assisted coding med MCP, model routing och repo-kontext i editorn. Representerar IDE-native agent trend snarare än ren chat API.
**GitHub Copilot API** — Code completion och chat i IDE via GitHub/Microsoft infra med repo policy. Enterprise controls för data som får skickas till modellen.
**MCP Server Ecosystem** — Community och vendor MCP servers för databaser, GitHub, Slack, filesystem m.m. Standardiserar tool access för agenter utan custom integration per tjänst.
**MCP Tool Discovery** — Runtime discovery av tillgängliga MCP tools via `tools/list` så agenten vet vad den kan anropa. Schema (JSON Schema) beskriver parametrar för LLM tool calling.
**MCP Resource URI** — Adresserbara datakällor (filer, docs) via MCP protocol med `resources/read`. Komplement till tools för kontext som agenten kan hämta selektivt.
**OpenAPI Tool Integration** — Exponera befintliga REST API som LLM tools genom att konvertera OpenAPI spec till function schema. Snabb väg att ge agents åtkomst till interna microservices.
**Plugin Marketplace** — Tredjepartsplugins för ChatGPT-liknande ekosystem (OAuth, actions). Föregångare till MCP; fortfarande relevant för consumer AI-plattformar.
**Model Card on Hub** — Metadata på HF: license, eval, intended use, limitations och training summary. Första stopp för due diligence innan download eller deploy.
**Model License** — Apache 2.0, MIT, Llama Community License, OpenRAIL m.fl. styr kommersiell användning, distillation och deployment. Licensbrott kan stoppa produktlaunch.
**Open Weights Model** — Public checkpoint med fri download (med licensvillkor); kan köras lokalt eller finjusteras. Motsats till API-only; shiftar säkerhets- och compliance-börda till deployer.
**API-Only Model** — Weights proprietary; endast inferens via provider API med ToS och logging. Enklare compliance för provider men vendor lock-in och data residency hos tredje part.
**Distillation from API Model** — Träna mindre modell på outputs från stor API-modell (teacher). Ofta förbjudet eller begränsat i ToS; juridiskt och etiskt gråzon utöver teknisk möjlighet.
**Terms of Service Constraints** — Juridiska begränsningar: no competitive models, no training on outputs, geographic restrictions, attribution. Läs ToS per provider innan produktbygge på API.
**Data Retention Policy** — Hur länge provider lagrar prompts, completions och logs; varierar per produkt och enterprise-avtal. Påverkar GDPR, HIPAA och om prompts får innehålla hemligheter.
**Zero Data Retention** — Provider lagrar inte customer data efter request (eller efter kort abuse window). Enterprise-funktion med ofta högre pris; verifiera vad "zero" exkluderar (billing metadata, abuse logs).
**Enterprise VPC Deployment** — Privat endpoint eller dedicated instance i kundens VPC så trafik inte går över public internet. Krävs i finans och reglerade branscher.
**Private Link** — Azure Private Link / AWS PrivateLink för connectivity till AI-tjänst utan public IP. Kombineras med VPC deployment och DNS privat zone.
**SOC 2 Compliance** — Säkerhetscertifiering (Type II) för enterprise SaaS: access control, logging, change management. Due diligence krav vid vendor-val av AI API.
**HIPAA BAA** — Business Associate Agreement som tillåter behandling av PHI via cloud AI under US healthcare rules. All data retention och subprocessors måste listas i BAA.
**GDPR Data Processing Agreement** — DPA för EU personuppgifter: processor/controller roller, sub-processors, SCC, rätt till radering. Obligatoriskt innan EU-kunddata skickas till US AI API.
**Regional API Endpoint** — EU/US/region-specifika base URLs för data residency och lägre latens. Pinna endpoint i SDK; fel region kan bryta compliance eller öka latency.
**Fallback Routing** — Byt automatiskt till backup provider eller modell vid 429, 5xx eller timeout. Kräver normaliserat prompt format och konsekvent kvalitet/test av backup.
**Cost Estimation API** — Beräkna token-kostnad före request via tokenizer eller provider pricing API. Viktigt för user-facing features med budget per session.
**Tokenizer API Utility** — Räkna tokens utan full inference (tiktoken, provider endpoint). Använd för pre-flight check av context limit och kostnad i UI.
**Playground UI** — Webb-UI hos provider för prompt testing, parameter tuning och shareable links utan kod. Bra för PM och red team; inte för prod secrets.
**SDK Python OpenAI** — Official Python client med sync/async, streaming, retries och typed helpers. Uppdateras vid API-ändringar; de facto referensimplementation.
**SDK Anthropic** — Official Python/TypeScript client för Claude Messages API med tool use och streaming. Parallell till OpenAI SDK i multi-provider appar.
**Community Leaderboard Ecosystem** — Open LLM Leaderboard, LMSYS Chatbot Arena, HuggingFace leaderboards för jämförelse av open och closed modeller. Driver transparens men också benchmark gaming.
**Spaces GPU Demo** — HF Spaces med Gradio/Streamlit och valfri GPU för interaktiva modell demos. Enkelt dela modell med community; inte production serving.
**Gradio Interface** — Snabb web UI för ML modeller med input widgets och API-liknande `/predict`. Populärt för demos och interna verktyg; begränsad auth och skalning.
**Streamlit App** — Python dashboard för AI prototypes och data apps med minimal frontend-kod. Bra för analytiker; prod kräver separat hosting och secrets-hantering.
**ComfyUI Workflow** — Node-based UI för diffusion pipelines med full kontroll över sampling, LoRA och conditioning. Power user-verktyg för bildgenerering; workflows delas som JSON.
**Automatic1111 WebUI** — Populärt Stable Diffusion web interface med extensions, inpainting och batch. Historiskt centralt open-source UI för diffusion community.
**Civitai Model Sharing** — Community hub för diffusion LoRAs, checkpoints och workflows med ratings och tags. Licens och NSFW-innehåll kräver egen moderation vid prod-användning.
**Replicate API** — Kör publika eller egna modeller som HTTP API med pay-per-second GPU. Snabb time-to-market utan egen GPU infra; cold start och kostnad vid hög QPS.
**Modal Serverless GPU** — Serverless functions med GPU för AI workloads (batch, fine-tune, inference) med scale-to-zero. Alternativ till egen K8s för sporadiska GPU-jobb.

## Matematik & Statistik

**Gradient** — Vektor av partiella derivator ∂L/∂θ som anger riktning och storlek för snabbaste ökning av förlusten i parameterutrymmet. I neurala nätverk beräknas den effektivt via kedjeregeln i backpropagation och styr varje viktuppdatering i SGD, Adam och andra optimizers.
**Jacobian Matrix** — Matris J där J_ij = ∂y_i/∂x_j som beskriver lokala linjära approximationer av en vektorfunktion y = f(x). Används i reverse-mode autodiff, stabilitetsanalys och när man propagerar osäkerhet genom sammansatta transformationer.
**Hessian Matrix** — Symmetrisk matris av andra derivator H_ij = ∂²L/∂θ_i∂θ_j som kvantifierar lokalkurvatur i förlustlandskapet. Används i Newton-metoder, analys av saddle points och för att bedöma om ett kritiskt punkt är minimum, maximum eller saddle.
**Chain Rule** — Kedjeregeln ∂L/∂x = (∂L/∂y)(∂y/∂x) som möjliggör gradientberäkning genom sammansatta funktioner lager för lager. Grunden i backpropagation: varje modul multiplicerar upstream-gradient med sin lokala Jacobian/transponerade Jacobian.
**Partial Derivative** — Derivata av en funktion med avseende på en variabel med alla övriga variabler hållna fixa, t.ex. ∂L/∂w_j. Byggsten i gradientvektorn och i partiell differentiering av multivariata förlustfunktioner.
**Total Derivative** — Summan av alla partiella effekter när variabler är implicit beroende, dL/dt = Σ_i (∂L/∂x_i)(dx_i/dt). Nödvändig när parametrar påverkar förlusten både direkt och via mellanliggande latent variabler.
**Gradient Descent Convergence** — Teoretiska villkor som L-smoothness och (strong) convexity som garanterar konvergens till globalt minimum med lämplig learning rate. I icke-konvexa deep learning ger dessa villkor ofta lokala garantier eller asymptotiska argument snarare än strikta globala bounds.
**Learning Rate Bound** — Teoretisk övre gräns η ≤ 2/L för konvergens under L-smoothness, där L är Lipschitz-konstanten för gradienten. Överskridande gränsen ger typiskt divergens eller oscillationer; praktisk tuning kombinerar denna teori med warmup och adaptiva scheman.
**Convex Function** — Funktion där f(λx+(1-λ)y) ≤ λf(x)+(1-λ)y för alla λ∈[0,1], vilket garanterar att varje lokalt minimum är globalt. Konvex optimering har effektiva algoritmer och tydliga konvergensgarantier, till skillnad från deep learning-förlustlandskap.
**Strong Convexity** — Strengthening av convexity: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||² med μ>0, vilket ger unikt minimum och linjär konvergenshastighet. Ger starkare generaliserings- och optimeringsgarantier än ren convexity men sällan exakt uppfyllt för neurala nätverk.
**Lipschitz Continuity** — Villkor |f(x)-f(y)| ≤ L||x-y|| som begränsar hur snabbt en funktion kan ändras; för gradienter innebär det ||∇f(x)-∇f(y)|| ≤ L||x-y||. Centralt antagande i konvergensteori och för att begränsa gradientexplosion.
**L-Smooth Function** — Funktion vars gradient är L-Lipschitz, ekvivalent med att Hessianens spektralnorm är ≤ L. Standardantagande i analys av gradient descent; styr tillåten learning rate och konvergenshastighet.
**PL Condition** — Polyak-Łojasiewicz: ||∇f(x)||² ≥ 2μ(f(x)-f*) som är svagare än strong convexity men ändå ger linjär konvergens till globalt minimum. Förklarar varför vissa icke-konvexa problem (t.ex. overparameteriserade nätverk) kan konvergera globalt.
**Stochastic Gradient Noise** — Varians i minibatch-gradienten σ² som introducerar brus i parameteruppdateringar jämfört med full-batch gradient. Bruset kan hjälpa generalisering genom implicit regularisering men saktar ner konvergens och kräver learning rate-anpassning.
**Variance Reduction** — Tekniker som SVRG, SAGA och adaptiva moment (Adam) som minskar gradientvarians utan att förlora SGD:s beräkningsfördelar. Minskar oscillationer och kan ge snabbare konvergens mot samma minimum som full-batch metoder.
**Expected Risk** — Förväntat förlustvärde E_{(x,y)~P}[L(f(x), y)] över den sanna datadistributionen P; det teoretiska generaliseringsmålet. Empirisk risk är ett Monte Carlo-estimat; gapet mellan dem kvantifieras av generaliseringsbounds.
**Empirical Risk** — Medelvärde av förlusten över träningsmängden (1/n)Σ L(f(x_i), y_i); det som faktiskt minimeras vid träning. Minimering av enbart empirisk risk utan regularisering riskerar overfitting när modellkapaciteten är hög.
**Generalization Bound** — Teoretisk probabilistisk gräns på |R_emp - R_exp| som funktion av datamängd n, modellkomplexitet och konfidensnivå δ. Exempel: bounds via VC-dimension, Rademacher complexity eller stabilitet som förklarar när låg träningsförlust implicerar låg testförlust.
**PAC-Bayes Bound** — Generaliseringsbound baserad på KL-divergens mellan posterior och prior över hypoteser, ofta skarpare än klassiska uniform bounds. Används teoretiskt för att motivera Bayesian och ensemble-metoder samt compression-baserad generalisering.
**Rademacher Complexity** — Mått på hypotesklassens rikedom via förväntad korrelation med slumpmässiga Rademacher-tecken ±1. Högre Rademacher complexity ⇒ större risk för overfitting; centralt i modern statistisk lärningsteori för att kvantifiera kapacitet.
**VC Dimension** — Största antal punkter som en hypotesklass kan klassificera godtyckligt (shatter); klassiskt kapacitetsmått i PAC-lärning. Växer ofta med antal parametrar men förklarar inte fullt deep learning där implicit regularisering spelar stor roll.
**Bias-Variance Decomposition** — Förväntat prediktionsfel kan skrivas som bias² + variance + irreducible noise (Bayes error). Hög bias = underfitting; hög variance = overfitting; modellval och regularisering balanserar dessa termer.
**Maximum Likelihood Estimation** — MLE väljer parametrar θ som maximerar sannolikheten (eller log-likelihood) för observerad data: θ̂ = argmax_θ Π p(x_i|θ). Asymptotiskt effektivt under standardregularity conditions; ekvivalent med att minimera negativ log-likelihood.
**Maximum A Posteriori** — MAP inför prior p(θ) och maximerar posterior: θ̂ = argmax_θ p(θ|D) ∝ p(D|θ)p(θ). Med Gaussian prior blir det ekvivalent med MLE plus L2-regularisering (weight decay); balanserar datat och prior-belief.
**Bayesian Inference** — Uppdatering av belief om parametrar via Bayes regel P(θ|D) ∝ P(D|θ)P(θ), vilket ger full posterior snarare än punkt-estimat. Ger naturlig osäkerhetsquantifiering men är ofta beräkningsmässigt intraktabelt för stora modeller.
**Posterior Distribution** — Sannolikhetsfördelning P(θ|data) efter att ha observerat evidens; sammanfattar kvarvarande osäkerhet om parametrar. Används för prediktion via marginalisering P(y|x,D) = ∫ P(y|x,θ)P(θ|D)dθ och för att kvantifiera epistemic uncertainty.
**Prior Distribution** — Belief P(θ) om parametrar före data ses; kodar induktiv bias (t.ex. sparsity, smoothness). Conjugate priors ger analytiska posteriors; i deep learning approximeras prior/posterior via weight decay eller variational inference.
**Conjugate Prior** — Prior som ger posterior i samma parametriserade familj, t.ex. Beta prior + Binomial likelihood → Beta posterior. Förenklar analytisk Bayesian uppdatering och möjliggör closed-form inferens utan numerisk integration.
**Variational Inference** — Approximera intractable posterior P(θ|D) med enklare variational family q_φ(θ) genom att maximera ELBO. Skalar till stora modeller (VAE, Bayesian neural nets) men introducerar approximationsfel från begränsad q-familj.
**ELBO** — Evidence Lower Bound: log p(x) ≥ E_q[log p(x|z)] - KL(q(z)||p(z)); det objektiv som maximeras i VAE och variational inference. Balanserar rekonstruktionskvalitet mot regularisering mot prior i latent space.
**Kullback-Leibler Divergence** — KL(P||Q) = E_P[log(P/Q)] mäter asymmetrisk "avstånd" mellan fördelningar; Q approximerar P. Används i VAE-regularisering, variational inference och knowledge distillation; asymmetri innebär att KL(P||Q) ≠ KL(Q||P).
**Cross-Entropy** — H(P,Q) = -Σ_x P(x) log Q(x) mäter expected coding length under Q när sann fördelning är P. Standard classification loss (softmax + CE) och language modeling loss; ekvivalent med negativ log-likelihood vid one-hot P.
**Mutual Information** — I(X;Y) = H(X) - H(X|Y) mäter mängd shared information mellan slumpvariabler; noll vid oberoende. Används i InfoNCE, feature selection, disentanglement och för att analysera vad representationer kodar.
**Entropy** — Shannon-entropi H(X) = -Σ p(x) log p(x) mäter osäkerhet eller expected information content i en fördelning. Maximal för uniform fördelning; låg entropi indikerar koncentrerad (förutsägbar) fördelning.
**Conditional Entropy** — H(X|Y) = E_Y[H(X|Y=y)] mäter kvarvarande osäkerhet i X givet Y. Relaterat till mutual information via I(X;Y) = H(X) - H(X|Y); centralt i informationsbottleneck-analys.
**Information Gain** — Entropiminskning ΔH = H(parent) - Σ (n_k/n) H(child_k) vid split i decision trees. Väljer attribut som maximalt reducerar label-osäkerhet; variant av mutual information mellan feature och target.
**Softmax Function** — σ(z)_i = exp(z_i)/Σ_j exp(z_j) mappar logits z till sannolikhetsvektor på simplexen (summa 1, alla ≥0). Differentierbar och konvex; numeriskt stabil med log-sum-exp-tricket; standard output-lager för multi-klass klassificering.
**Log-Sum-Exp Trick** — Beräknar log(Σ exp(x_i)) som m + log(Σ exp(x_i - m)) med m = max(x_i) för att undvika overflow/underflow. Kritiskt för stabil softmax, cross-entropy och log-partition functions i probabilistiska modeller.
**Sigmoid Function** — σ(x) = 1/(1+e^{-x}) mappar reella tal till (0,1); används i binär klassificering, BCE loss och gating (LSTM, GLU). Lider av vanishing gradient vid extrem input men ger tolkbar sannolikhet per klass.
**ReLU Derivative** — ReLU'(x) = 1 om x>0, annars 0; subgradient vid x=0 (ofta satt till 0 eller 1). Enkel och snabb; döda neuroner uppstår när x≤0 permanent eftersom gradienten blir noll.
**Softplus** — Softplus(x) = log(1+exp(x)) är en mjuk, differentierbar approximation av ReLU med överallt positiv gradient. Används när strikt differentiability krävs; närmar sig ReLU asymptotiskt för stora |x|.
**Gaussian Distribution** — N(μ,σ²) med PDF ∝ exp(-(x-μ)²/(2σ²)); central i statistik och ML som noise model, prior och initiering. CLT motiverar normalapproximationer; multivariat generalisering använder mean μ och covariance Σ.
**Multivariate Gaussian** — N(μ, Σ) med density ∝ exp(-½(x-μ)ᵀΣ⁻¹(x-μ)); modellerar vektorvalda slumpvariabler med korrelationer. Central i GMM, Kalman filter, FID-metrik och som prior/posterior i linjära Gaussian modeller.
**Covariance Matrix** — Σ_ij = Cov(X_i, X_j) = E[(X_i-μ_i)(X_j-μ_j)] beskriver varians längs diagonalen och korrelation mellan komponenter. Måste vara positiv semidefiniv; estimeras från data för PCA, whitening och multivariat analys.
**Precision Matrix** — Precision Λ = Σ⁻¹; off-diagonal Λ_ij ≠ 0 indikerar conditional dependence givet övriga variabler (Markov properties i Gaussian graphical models). Nolltermer i precision motsvarar conditional independence i Gaussian MRF.
**Central Limit Theorem** — Summan (eller medelvärdet) av i.i.d. variabler med ändlig varians konvergerar i fördelning till N(0,1) efter standardisering. Motiverar normalapproximationer av noise, gradient noise och aggregerade metrics vid stora n.
**Law of Large Numbers** — Sample mean X̄_n konvergerar mot E[X] när n→∞ (svagt eller starkt LLN under i.i.d.-antaganden). Motiverar att empirisk risk approximerar expected risk och att Monte Carlo-estimat förbättras med fler samples.
**Markov Chain** — Stokastisk process där P(X_{t+1}|X_t, X_{t-1},...) = P(X_{t+1}|X_t); framtiden beror endast på nuvarande state. Grund för MCMC, Markov modeller i NLP (HMM) och diskreta state transitions i reinforcement learning.
**Stationary Distribution** — Fördelning π som uppfyller π = πP (eller πP = π för radvektor) så att kedjan inte ändrar marginal efter transition. Långsiktig steady state under ergodicitet; avgör MCMC-samples och PageRank-liknande processer.
**Detailed Balance** — Tillstånd π(i)P(i→j) = π(j)P(j→i) är sufficient (men inte nödvändigt) för stationarity. Används i design av Metropolis-Hastings och Gibbs sampling för att garantera korrekt target-fördelning.
**Monte Carlo Estimation** — Approximera expectation E[f(X)] med sample mean (1/n)Σ f(X_i) där X_i ~ P. Konvergens O(1/√n); grund för stochastic gradient, policy gradient och Bayesian MCMC-estimat.
**Importance Sampling** — Estimera E_p[f(X)] via samples från q med vikter p(x)/q(x): (1/n)Σ f(X_i) w(X_i). Minskar varians om q närmar sig p; centralt i off-policy RL, rare event simulation och variational bounds.
**Markov Chain Monte Carlo** — MCMC genererar korrelerade samples från komplex posterior via Markov transitions (Metropolis-Hastings, Gibbs, HMC). Konvergerar till target-fördelning asymptotiskt; används i Bayesian inference när closed-form posterior saknas.
**Eigenvalue Decomposition** — För symmetrisk A: A = QΛQᵀ där Q är ortogonal och Λ diagonal med egenvärden. PCA projicerar på egenvektorer med störst egenvärden; spektralnorm och condition number härleds från egenvärden.
**Singular Value Decomposition** — SVD: A = UΣVᵀ med singulära värden σ_i på diagonalen i Σ; gäller alla reella matriser. Används för lågranksapproximation, pseudoinvers, numerisk stabilitet och analys av weight matrices i deep learning.
**Matrix Rank** — Rank = dimension av kolumn- (eller rad-)rymd = antal linjärt oberoende kolumner. Lågrank struktur möjliggör komprimering (SVD-truncation) och indikerar redundans; rank deficiency kan orsaka icke-unika lösningar.
**Condition Number** — κ(A) = σ_max/σ_min (för SVD) mäter känslighet för numeriska fel i linjära system Ax=b. Högt κ ⇒ ill-conditioned; små perturbationer i data ger stora ändringar i lösning — problematiskt vid inversion och optimering.
**Positive Definite Matrix** — Symmetrisk A med xᵀAx > 0 ∀x≠0; alla egenvärden > 0. Covariance-matriser och Hessianer vid strikta lokala minima är PD; PD säkerställer att Cholesky-dekomposition existerar och att kvadratiska former är konvexa nedåt.
**Dot Product** — Skalarprodukt x·y = Σ x_i y_i mäter alignment och projektion; i ML används för attention scores (Q·Kᵀ), linear layers och cosine similarity efter normalisering. Geometriskt: ||x|| ||y|| cos θ.
**Vector Norm** — L2-norm ||x||₂ = √(Σ x_i²) mäter euklidiskt avstånd; L1-norm Σ|x_i| inducerar sparsity vid regularisering; L∞ max|x_i|. Normval påverkar optimering, regularisering och avståndsmått i embedding space.
**Cosine Similarity Math** — cos θ = (x·y)/(||x|| ||y||) mäter vinkel mellan vektorer, invariant mot skalning. Standard i retrieval, semantic search och kontrastiv learning där riktning (semantik) viktigare än magnitud.
**Orthogonal Matrix** — Q med QᵀQ = QQᵀ = I; kolumner är ortonormala. Rotationer och reflectioner bevarar norm och vinklar; används i QR-dekomposition, orthogonal init och Muon optimizer (orthogonalized updates).
**Projection Matrix** — P = A(AᵀA)⁻¹Aᵀ projicerar vektorer till kolumnrummet för A (minsta kvadrat-lösning). Idempotent (P²=P); central i linjär regression, PCA-rekonstruktion och subspace-metoder.
**Lagrange Multipliers** — Metod för constrained optimization: inför multiplikatorer λ så att ∇f = Σ λ_i ∇g_i vid optimum under constraints g_i(x)=0. Förklarar dual variables och KKT-villkor; dual problem ger bounds på primal optimum.
**Karush-Kuhn-Tucker Conditions** — KKT: nödvändiga (och under convexity, tillräckliga) villkor för optimum under jämlikhets- och olikhetsconstraints. Inkluderar stationarity, primal/dual feasibility och complementary slackness; grund för SVM dual och constrained deep learning.
**Convex Optimization** — Minimera konvex f över konvex mängd; varje lokalt minimum är globalt och dualitet ger kraftfulla verktyg. LP, QP och SOCP är specialfall; de flesta deep learning-problem är icke-konvexa men delproblem (t.ex. SVM dual) är konvexa.
**Linear Programming** — Optimera lineärt objektiv cᵀx under lineära constraints Ax≤b; optimum vid hörn av polytop. Simplex och interior-point metoder; används i resource allocation, transport och som relaxation av kombinatoriska problem.
**Quadratic Programming** — Minimera ½xᵀQx + cᵀx under lineära constraints med Q positiv semidefiniv. SVM dual form är QP; effektiva algoritmer för medelstora problem; generaliserar linjär till margin-maximering med kernel trick.
**Stochastic Process** — Familj av slumpvariabler {X_t} indexerade över tid eller space; modellerar dynamik med osäkerhet. Brownian motion, Poisson processer och diffusion SDE:er är centrala i score-based generative models och finansiell matematik.
**Brownian Motion** — Wiener process W_t med oberoende stationära inkrement W_{t+s}-W_t ~ N(0,s) och kontinuerliga paths. Matematisk grund för SDE diffusion term g dW; self-similar och Gaussian; används i Black-Scholes och DDPM forward process.
**Stochastic Differential Equation** — SDE dX_t = f(X_t,t)dt + g(X_t,t)dW_t beskriver dynamik med deterministisk drift f och stokastisk diffusion g. Score-based generative models lär reverse SDE/ODE för sampling; kräver Itô- eller Stratonovich-kalkyl.
**Itô's Lemma** — Kedjeregel för Itô-processer: om dX = μ dt + σ dW då dF = (∂F/∂t + μ∂F/∂x + ½σ²∂²F/∂x²)dt + σ∂F/∂x dW. Extra ½σ²∂²F/∂x²-term jämfört med vanlig kedjeregel; central i diffusion model-derivationer.
**Fokker-Planck Equation** — PDE ∂p/∂t = -∇·(fp) + ½∇²(g²p) beskriver tidsutveckling av densitet p(x,t) under SDE. Kopplar forward diffusion till reverse score; analytisk bridge mellan stokastisk dynamik och deterministisk flödesformulering.
**Wasserstein Distance** — W_p(P,Q) = (inf_{γ∈Γ(P,Q)} E_{(x,y)~γ}[||x-y||^p])^{1/p}, "earth mover's distance" mellan fördelningar. Metrik (symmetrisk, triangle inequality); WGAN använder W_1 med Lipschitz constraint; känsligare än KL för disjoint supports.
**Total Variation Distance** — TV(P,Q) = max_A |P(A)-Q(A)| = ½||P-Q||_1; starkt mått på fördelningsskillnad. Används i konvergensanalys av MCMC, mixing time och privacy bounds; max över alla mätbara mängder A.
**Jensen's Inequality** — För konvex f: f(E[X]) ≤ E[f(X)]; för konkav inverteras inequality. Motiverar variational bounds (ELBO), konvexity av log-sum-exp och att expected risk ≥ risk of expected predictor under konvex loss.
**Chebyshev's Inequality** — P(|X-μ| ≥ kσ) ≤ 1/k² för vilken fördelning som helst med ändlig varians. Grov men universell probabilistisk bound; grund för weak LLN och konfidensargument utan distributional antaganden.
**Hoeffding Bound** — Exponential tail bound för summa av oberoende bounded variabler: P(|S_n - E[S_n]| ≥ t) ≤ 2exp(-2t²/n). Skarpare än Chebyshev för bounded data; används i PAC bounds och koncentrationsargument i ML-teori.
**Confidence Interval** — Intervall [L, U] som med sannolikhet ≥ 1-α innehåller sann parameter θ; konstrueras från sample statistics och fördelningsantaganden. Bootstrap och normal approximation vanliga; skilj från prediction interval som täcker nya observationer.
**Hypothesis Testing** — Formell ram för H₀ (null) vs H₁ (alternativ): beräkna teststatistik, p-value och besluta vid signifikansnivå α. Kontrollerar Type I/II-fel; central i A/B-test, benchmark-signifikans och reproducerbarhetsbedömning.
**p-value** — Sannolikhet att observera data minst lika extrem som observerat under antagande att H₀ är sann. Lågt p-value ⇒ evidence mot H₀; tolkas inte som sannolikhet att H₀ är sann; påverkas av sample size och multiple testing.
**Type I Error** — Falskt positiv: reject H₀ när H₀ faktiskt är sann; sannolikhet α (signifikansnivå). I ML-experiment kan det innebära att en "signifikant" metric-förbättring är slumpmässig noise.
**Type II Error** — Falskt negativ: fail to reject H₀ när H₁ är sann; sannolikhet β. Komplement till statistical power (1-β); låg power ⇒ missar verkliga effekter trots tillräcklig metod.
**Statistical Power** — Sannolikhet 1-β att korrekt reject H₀ när H₁ är sann. Ökas med större sample, större effektstorlek och lägre varians; viktigt vid design av benchmarks och kliniska/ experimentella studier.
**Bayes Factor** — BF₁₀ = p(data|H₁)/p(data|H₀), ratio av marginal likelihoods; quantifierar evidence för H₁ vs H₀. Skala (Jeffreys): BF>10 stark evidence; integrerar över parametrar under varje hypotes, till skillnad från p-value.
**Causal Inference** — Metoder för att inferera kausal effekt P(Y|do(X)) snarare än associativ P(Y|X); kräver strukturella antaganden eller design. do-calculus, instrumental variables, DAGs och RCT:er adresserar confounding som korrelation inte kan lösa.
**Confounding Variable** — Z som påverkar både behandling X och outcome Y, skapar spurious correlation X-Y om Z inte kontrolleras. Naiv regression ger biased causal estimate; löses via randomisering, conditioning, matching eller IV.
**Randomized Controlled Trial** — RCT randomiserar enheter till treatment/control så att förväntad confounding elimineras; gold standard för kausal effekt av intervention. A/B-test i tech är RCT; etiska och praktiska begränsningar i vissa domäner.
**Law of Total Expectation** — Tower property: E[X] = E[E[X|Y]]; marginalisering via conditioning på latent eller grupperande variabel Y. Centralt i EM-algoritm, hierarchical models och för att dekomponera förväntningar i deep generative modeller.
**Law of Total Variance** — Var(X) = E[Var(X|Y)] + Var(E[X|Y]): total varians = within-group varians + between-group varians. Används i ANOVA, mixed models och för att förstå hur latent struktur bidrar till observerad variabilitet.
## Forskningskoncept

**Ablation Study** — Systematiskt ta bort eller ersätta en komponent i modell/pipeline (t.ex. attention head, loss term) och mäta prestandaförändring. Isolerar varje delens bidrag och avslöjar vilka designval som faktiskt driver resultat versus är redundant.
**Baseline Comparison** — Jämför föreslagen metod mot etablerad baseline under identiska data splits, hyperparametrar och compute budget. Utan rättvis jämförelse är claimed improvements svåra att tolka; baselines ska vara starka och reproducerbara.
**State of the Art** — SOTA: bästa publikt rapporterade resultat på given benchmark vid en tidpunkt, ofta leaderboard-topp. SOTA är flyktigt och dataset-specifikt; små metric-gains kan bero på tuning snarare än fundamental novelty.
**Reproducibility Crisis** — Många ML-resultat svåra att replikera p.g.a. saknad kod, data, seeds eller dolda hyperparametrar. Undergräver tillit till litteraturen; NeurIPS m.fl. inför reproducibility checklists och open-source-krav.
**Replication Study** — Oberoende team kör samma experiment med samma protokoll för att verifiera ursprungliga claims. Misslyckad replication kan avslöja implementation bugs, cherry-picking eller överdriven generalisering av resultat.
**Pre-registration** — Publicera hypotes, metod och analysplan före datainsamling/experiment för att binda forskaren till protokollet. Motverkar p-hacking och HARKing (hypothesizing after results are known); vanligare i medicin än ML men växande.
**Peer Review** — Expertgranskning av metod, novelty, validitet och presentation före publicering. Double-blind minskar bias; begränsningar inkluderar inconsistent quality, reviewer bandwidth och svårighet att bedöma stora compute-experiment.
**Open Review** — Public reviews och författarsvar (t.ex. OpenReview.net) ökar transparens i beslutsprocessen. Möjliggör community-granskning men kan avskräcka kritik eller leda till för tidig idéexponering.
**ArXiv Preprint** — Early dissemination på arXiv före (eller parallellt med) peer review; accelererar kunskapsspridning. Preprint ≠ peer-reviewed; versionering och citation av rätt version viktigt.
**Conference vs Journal** — Konferenser (NeurIPS, ICML) har snabb cykel och hög synlighet; journaler (JMLR) längre granskning och mer utrymme. ML-fältet prioriterar konferenser historiskt; journal-publicering ger permanent arkiv och djupare revision.
**NeurIPS** — Neural Information Processing Systems; premier ML/AI-konferens med tusentals submissions och låg accept rate. Benchmark för cutting-edge forskning inkl. deep learning, RL, generative models och alignment.
**ICML** — International Conference on Machine Learning; en av de tre toppkonferenserna tillsammans med NeurIPS och ICLR. Täcker brett ML-teori och tillämpningar med rigorös review-process.
**ICLR** — International Conference on Learning Representations; fokus på representation learning och deep learning. OpenReview-baserad review; stark närvaro av transformer, generative models och interpretability-forskning.
**ACL EMNLP** — Association for Computational Linguistics och Empirical Methods in NLP; ledande venues för språkteknologi. Täcker parsing, MT, LLM-evaluering, multilingual NLP och discourse; empirisk validering central.
**CVPR ICCV** — Computer Vision and Pattern Recognition (årlig) och International Conference on Computer Vision (varannan); toppvenues för bildförståelse, detection, segmentation och generativ vision.
**Spotlight vs Oral** — Högt rankade papers får spotlight (kort presentation + poster) eller oral (längre session); signalerar committee-prioritering. Oral papers får mer synlighet men spotlight erbjuder fortfarande hög kvalitetsmarkering.
**Best Paper Award** — Utmärkelse för exceptionell novelty, rigor och potentiell impact enligt program committee. Ökar citation och synlighet; ibland kontroversiellt om val speglar hype mer än långsiktig betydelse.
**Workshop Paper** — Mindre venue (ofta colocated med konferens) för tidiga idéer, niche topics och community-building. Lägre review bar men värdefullt för feedback; räknas sällan som full publication i tenure-track sammanhang.
**Technical Report** — Institutionell rapport (Google, Meta, OpenAI) utan full peer review; kan vara omfattande och snabb. Stor impact (Transformer, GPT-serien) men saknar formell validering; versionering och intern review varierar.
**Novelty Claim** — Påstående om ny metod, teori eller empirisk insikt som skiljer paper från prior art. Måste stödjas av thorough related work; överdriven novelty riskerar desk rejection eller replication failure.
**Related Work Section** — Kontextualiserar bidrag mot prior art: vad finns, vad saknas, hur skiljer sig detta arbete. Visar att författare känner fältet; svag related work underminerar novelty-claims och review-förtroende.
**Contribution Statement** — Explicit lista (ofta punktlista) över paperets nya bidrag: metod, teori, dataset, benchmark. Hjälper reviewers bedöma scope; bör vara specifik och verifierbar, inte vag "we propose a novel framework".
**Limitations Section** — Erkänner svagheter: dataset bias, compute constraints, generaliseringsgränser, ethical risks. Ökar trovärdighet och hjälper framtida arbete; numera ofta required av konferenser.
**Broader Impact Statement** — Diskussion av etiska, samhälleliga och miljömässiga konsekvenser av forskningen. NeurIPS m.fl. kräver detta; tvingar reflektion kring dual use, bias och energiförbrukning.
**Compute Reporting** — Dokumentera GPU-typ, antal GPU-timmar, energi och carbon footprint för experiment. NeurIPS reproducibility checklist; möjliggör rättvis jämförelse och hållbarhetsdebatt kring storskalig ML.
**Hyperparameter Table** — Full lista HP (LR, batch, epochs, weight decay, etc.) i main text eller appendix för reproduktion. Saknade HP är vanligaste reproducibility-felet; inkludera sökintervall om tuning gjordes.
**Appendix Details** — Arkitekturspecifikationer, fullständiga prompts, extra experiments och proofs som inte får plats i main paper. Ofta där reproducerbarhet faktiskt lever; reviewers kollar sällan appendix noggrant.
**Supplementary Material** — Extra figurer, videos, kod-länkar och data sheets utöver appendix. Kan inkludera interactive demos; kvalitet varierar; kod-release starkt korrelerar med successful replication.
**Anonymous Submission** — Double-blind review döljer författarnamn och institution för att minska prestige bias. Kräver anonymiserad kod/data och undvikande av self-citation som avslöjar identitet; svårt med preprint-arXiv.
**Single-Blind Review** — Reviewer anonym men författare känd; vanligt i vissa journaler. Reviewer kan vara friare i kritik men riskerar power imbalance och retaliation concerns (sällan i praktiken).
**Author Response Period** — Rebuttal-fas där författare svarar på reviewer-kommentarer före final decision. Area chair väger rebuttal mot reviews; effektiv rebuttal kan vända borderline reject till accept.
**Meta-Review** — Area chair (senior meta-reviewer) syntetiserar individuella reviews till accept/reject-beslut med motivering. Balanserar motstridiga opinions; meta-review synlig i OpenReview och förklarar final outcome.
**Accept Rate** — Andel submitted papers som accepteras; NeurIPS/ICML ofta ~20-25%. Låg rate ⇒ hög konkurrens men imperfect proxy för kvalitet; accept rate varierar per track och år.
**Citation Count** — Antal gånger paper citeras; proxy för impact men laggad (år), fältberoende och gameable. H-index och citation counts favoriserar etablerade fält och engelskspråkig litteratur; använd med försiktighet.
**h-index** — Forskare har h-index h om h papers har ≥ h citations vardera. Kombinerar produktivitet och impact; missar enskilda högciterade papers och unga forskare; populär i akademisk utvärdering trots kända brister.
**Impact Factor** — Journal metric: genomsnittliga citations per paper över tidsfönster; kontroversiell kvalitetsproxy. ML-fältet värderar konferenser högre; IF favoriserar review-articles och etablerade journaler.
**Seminal Paper** — Grundläggande arbete som definierar eller transformerar ett fält: Transformer (2017), ResNet, AlexNet, diffusion models. Hög citation, många follow-ups; ofta blir standard baseline eller arkitektur att bygga vidare på.
**Follow-Up Work** — Förbättringar, extensions och kritiska analyser av tidigare metod. Healthy ecosystem: seminal paper → ablations → scaling → mechanistic understanding; distinkt från concurrent discovery.
**Concurrent Discovery** — Oberoende teams publicerar liknande idéer samtidigt (t.ex. diffusion, GAN variants). Visar att idéer är "in the air"; priority disputes och arXiv timestamps blir relevanta.
**Negative Results** — Rapportera att hypotes eller metod inte fungerade som förväntat; underskattat publiceringsvärde. Viktigt för att undvika duplicated effort; workshops och journals för negative results växer långsamt.
**Null Result Publication** — Publicera när experiment inte bekräftar hypotes; motverkar publication bias. Sällsynt i ML där positiva SOTA-gains prioriteras; viktigt för meta-analys och metodologisk ärlighet.
**Benchmark Overfitting** — Metod optimerad (medvetet eller omedvetet) mot specifik test suite via repeated eval, HP tuning på test eller data leakage. Leaderboard-topp ≠ generalisering; hold-out test och fresh benchmarks mitigerar.
**Leaderboard Chasing** — Forskning driven av små metric-gains (0.1 BLEU, 0.5 F1) utan djupare insikt eller ny förståelse. Kan accelerera progress men riskerar brittle methods och neglect av robusthet/fairness.
**Scaling Study** — Systematisk variation av modellstorlek, datamängd och compute för att kartlägga beteende. Empirisk grund för scaling laws; kräver loggar över flera orders of magnitude och kontrollerade jämförelser.
**Scaling Law Paper** — Empiriska potenslagar: loss ∝ N^(-α), D^(-β), C^(-γ) för params N, data D, compute C. Chinchilla, Kaplan et al. styr pretrain budget-allokering; laws är approximativa och dataset-beroende.
**Emergent Phenomena Paper** — Dokumenterar förmågor (arithmetic, ICL, chain-of-thought) som plöttligt dyker upp vid skalning. Kontrovers kring metric artifacts vs genuine emergence; viktigt för capability forecasting och safety.
**Mechanistic Interpretability** — Förstå vilka interna circuits, features och attention patterns som implementerar specifikt beteende. Reverse-engineering neural networks; tools: activation patching, sparse autoencoders, circuit discovery.
**Circuits Analysis** — Identifiera minimal subgrafer (heads, MLP neurons) ansvariga för specifik funktion (t.ex. indirect object identification). Hypothesis-driven dissection; kombinerar ablation, causal tracing och feature visualization.
**Probing Classifiers** — Tränade klassificerare på frozen interna representationer för att testa om information (syntax, POS) är linearly decodable. Enkel probe ⇒ information present; debatten: encoded vs used by model.
**Linear Probing** — Enkel logistic/linear regression på frozen layer outputs; standard probing baseline. Minimal probe capacity ⇒ starkare evidence att information är linearly structured i representationen.
**Representation Similarity** — CKA, SVCCA, Procrustes jämför interna representationer mellan modeller, lager eller seeds. Hög similarity ⇒ convergent representations; används för model merging och understanding training dynamics.
**CKA Centered Kernel Alignment** — Mått på representation similarity invariant till orthogonal transformation and isotropic scaling. Populärt för comparing layers across architectures; debatterat om det mäter meaningful alignment.
**Transfer Learning Study** — Pretrain på source task A, fine-tune/eval på target B; mät transfer efficiency och negative transfer. Kartlägger vilka representationer generaliserar; central för foundation model evaluation.
**Zero-Shot Transfer** — Eval på ny uppgift utan fine-tuning; modellen förlitar sig på pretrain + prompting. LLM-capability benchmark; skilj från zero-shot i classical CV där features transfereras implicit.
**Few-Shot Learning Research** — Lär från få labeled exempel via meta-learning, metric learning eller in-context learning. Omniglot, miniImageNet benchmarks; LLM few-shot via prompts har omdefinierat fältet.
**Meta-Learning** — "Learn to learn": optimera meta-parameters så inner-loop adaptation är snabb på nya tasks. MAML, Prototypical Networks; outer loop över task distribution; relaterat till LLM pretraining as meta-learning.
**MAML** — Model-Agnostic Meta-Learning: gradient-baserad meta-learning med few-step inner loop adaptation. Initialisering nära många tasks' optima; computationellt dyrt (second-order); inspiration för adaptation research.
**In-Context Learning Research** — Studera ICL som implicit Bayesian inference, gradient descent eller retrieval från pretrain. Mechanistic papers på induction heads; förklarar varför LLMs lär nya tasks från exemplar i prompt.
**Inductive Bias Analysis** — Vilka strukturer och lösningar arkitektur och optimizer prioriterar (t.ex. CNN → translation equivariance, transformer → pairwise interactions). Förklarar generalization beyond capacity arguments.
**Universal Approximation Study** — Teoretiska resultat att neurala nät (MLP, etc.) kan approximera kontinuerliga funktioner godtyckligt väl med tillräcklig bredd/djup. Garantiar existens, inte learnability, sample complexity eller effektivitet.
**Sample Complexity Analysis** — Antal exempel n krävs för ε-optimal generalization med sannolikhet 1-δ; bounds via VC, Rademacher, margin. Teoretisk complement till empirisk scaling; ofta loose för deep nets men vägledande.
**Optimization Landscape** — Geometri av loss surface: local minima, saddle points, flat vs sharp minima, connectivity. Visualisering och theory förklarar varför SGD hittar bra lösningar i höga dimensioner.
**Loss Landscape Visualization** — 2D/1D slices (filter normalization), mode connectivity plots och Hessian spectra. Intuition för training dynamics; filter-wise normalization nödvändig för meningsfull comparison.
**Mode Connectivity** — Olika trained minima kan kopplas via paths i weight space med nästan konstant låg loss. Implikerar att loss landscape har connected low-loss regions; relevant för ensembling och model merging.
**Lottery Ticket Hypothesis** — Dense network innehåller sparse subnetwork ("winning ticket") tränbar isolerat till samma prestanda från samma init. Sparsity + early rewind; öppnar frågor om overparameterization och pruning.
**Double Descent Phenomenon** — Test error minskar, ökar vid interpolation threshold, sedan minskar igen med modellstorlek. Utmanar klassisk bias-variance U-curve; kopplat till overparameterization och implicit regularization.
**Grokking** — Delayed generalization: perfect training fit länge före test generalization på algorithmic tasks (modular arithmetic). Tyder på att strukturell förståelse kan emerge efter memorization; active research area.
**Benign Overfitting** — Modell interpolerar träningsdata (noll training error) men generaliserar ändå väl på test. Overparameterization + implicit regularization; theory via minimum norm interpolators och alignment.
**Implicit Regularization** — SGD (och batch norm, early stopping) inducerar bias mot enkla/lågnorm lösningar utan explicit penalty. Förklarar delvis varför overparameterized nets inte alltid overfitar; LR och batch size påverkar implicit bias.
**Neural Tangent Kernel** — NTK: infinite-width limit där training blir kernel regression med fixed kernel Θ. Lazy regime: weights stay near init; analytisk training dynamics men begränsad feature learning.
**Lazy Training Regime** — Nätverk nära initiering; representationer ändras lite; NTK approximation gäller. Stor width, liten LR; kontrast till rich/feature learning regime där representations evolve substantially.
**Feature Learning Regime** — Aktiva representation changes under träning; rikare än NTK lazy limit. Praktisk deep learning opererar här; finite-width effects och hierarchical feature learning centrala.
**Scaling Hypothesis** — Testbar förutsägelse om beteende vid större scale (capability, loss, emergence). Popperian science i ML: scale up → confirm/refute; styr investment i compute och data collection.
**Bitter Lesson** — Sutton: general methods leveraging massive compute slår hand-engineered domain knowledge historiskt. Argument för scaling, search and learning över feature engineering; kontroversiellt för data-efficiency och safety.
**Research Debt** — Accumulerad komplexitet, undocumented conventions och tribal knowledge som saktar nycomers. Distillation papers och open cookbooks (e.g. training LLMs) adresserar debt; växer med field velocity.
**Benchmark Contamination Study** — Kvantifiera overlap mellan pretrain data och benchmark test sets via n-gram matching, embedding similarity. Contamination inflate metrics; critical för honest LLM eval (e.g. MMLU, GSM8K checks).
**Data Provenance Research** — Spåra ursprung, licens, consent och sammansättning av träningsdata. Data Statements, Datasheets for Datasets; legal/ethical compliance och reproducibility kräver provenance.
**Synthetic Data Research** — Kvalitet, diversity och begränsningar av AI-genererad träningsdata; risk för model collapse. När synthetic data hjälper (augmentation, rare classes) vs skadar (homogenization, error accumulation).
**Multimodal Research Trend** — Unified models över text, bild, ljud, video (GPT-4V, Gemini, Flamingo). Joint embedding spaces, any-to-any generation; evaluation harder than unimodal; data pairing challenges.
**Agent Research Frontier** — Autonomous tool-using LLM systems: web browsing, code execution, multi-step planning. SWE-bench, WebArena benchmarks; safety, reliability och long-horizon credit assignment open problems.
**Reasoning Research** — Process supervision, outcome reward models, verifiers, test-time compute scaling (o1-style). Chain-of-thought, tree search, formal verification; skilj genuine reasoning från pattern matching.
**Open Science Movement** — Open weights, datasets, eval harnesses och reproducibility standards (MLCommons, BigScience). Tension med safety (open weights misuse) och commercial interests (closed models).
**Collaborative Research** — Stora multi-institution teams (BigScience, LAION) och industry-academia partnerships. Diverse expertise och compute pooling; authorship, credit assignment och coordination overhead.
**Industry Lab Publication** — Google DeepMind, OpenAI, Meta FAIR driver stor del av frontier research med massive compute. Fast release cycles; tension mellan publication, productization och competitive advantage.
**Academic-Industry Gap** — Skillnad i compute access, data access, publication incentives och talent flow. Academia: theory, safety, niche benchmarks; industry: scaling, product; partnerships bridge gap.
**Patent vs Publication** — IP-skydd via patents vs open dissemination via papers; tradeoff for commercial labs. Patents delay disclosure; publications build reputation; some methods patented after paper (attention mechanisms debated).
**Research Ethics Approval** — IRB/ethics board för human subjects data, surveys, biometric data och deceptive studies. GDPR, consent forms; ML increasingly needs ethics review for human data and deployment studies.
**Dual Submission Policy** — Konferenser förbjuder samtidig submit till multiple venues med overlapping content. Violation → desk reject eller retraction; arXiv preprint usually allowed with disclosure.
**Citation Ethics** — Korrekt attribut till prior work; undvik plagiarism, citation cartels och strategic omission av competing methods. Ethical citation inkluderar self-citation transparency och crediting datasets, codebases och foundational ideas; undermining trust in scholarly record.
**P-hacking Awareness** — Cherry-pick metrics, seeds, subsets eller stopping time tills signifikant resultat uppstår. Multiple comparison problem; pre-registration, held-out test och honest reporting motverkar; widespread risk in competitive benchmarks.
## NLP & Text

**Natural Language Processing** — NLP: automatisk förståelse, generering, transformation och analys av naturligt språk med algoritmer och modeller. Spänner från klassisk pipeline (tokenize → parse → semantics) till end-to-end neurala och LLM-baserade system för nästan alla textuppgifter.
**Tokenization Strategy** — Val av subword-algoritm (BPE, WordPiece, Unigram, byte-level) påverkar OOV-hantering, sekvenslängd, morfologi och flerspråkighet. Fel tokenization kan fragmentera sällsynta ord och öka sekvenslängd; måste vara konsekvent mellan träning och inferens.
**WordPiece Tokenization** — Greedy subword merge-baserad tokenization använd i BERT; continuation tokens prefixas med ##. Balanserar ord- och subword-nivå; effektiv för engelska men kan vara suboptimalt för agglutinerande språk.
**Unigram Language Model Tokenizer** — SentencePiece Unigram: probabilistiskt urval av subword vocab genom att successivt ta bort tokens som minst försämrar LM-likelihood. Ofta bättre kompression än BPE; language-agnostic utan explicit whitespace rules.
**Byte-Level BPE** — BPE opererar på bytes/UTF-8 bytes istället för tecken; GPT-2/3/4 approach. Robust för alla Unicode utan UNK-token; längre sekvenser för icke-latinska skript men universell täckning.
**Part-of-Speech Tagging** — Tilldelar grammatisk kategori (noun, verb, adj, etc.) per token; grundläggande syntaktisk annotation. Historiskt HMM/CRF; nu oftast biLSTM-transformer eller end-to-end LLM; viktigt för downstream parsing och IE.
**Named Entity Recognition** — NER: identifierar och klassificerar entiteter (PER, ORG, LOC, MISC) som spans i text. Sequence labeling med BIO-schema; används i knowledge extraction, search och compliance; LLMs gör zero-shot NER via prompting.
**Dependency Parsing** — Analyserar grammatiska beroenden (head-dependent relations) mellan ord, bildar trädstruktur. Transition-based eller graph-based parsers; UD universal dependencies standard; viktigt för semantisk analys och relation extraction.
**Constituency Parsing** — Bygger phrase structure tree med fraser (NP, VP, PP) enligt context-free grammatik. CKY, neural chart parsers; mer struktur än dependencies; används i lingvistik och vissa generation pipelines.
**Semantic Role Labeling** — SRL: identifierar who did what to whom (Agent, Theme, Instrument) per predicate. Shallow semantic analysis; PropBank/Nombank frames; hjälper question answering och event extraction.
**Coreference Resolution** — Kopplar pronomen, definita beskrivningar och namn till samma entitet i discourse. Clustering eller end-to-end neural (e.g. SpanBERT); kritisk för document understanding och sammanhängande sammanfattning.
**Word Sense Disambiguation** — Välj rätt betydelse av homonym/polysem ord givet kontext (bank = financial vs river). WordNet senses; supervised och knowledge-based metoder; delvis löst implicit av contextual embeddings och LLMs.
**Machine Translation** — Automatisk översättning mellan naturliga språk; historiskt phrase-based SMT, nu dominerat av neural MT. Utvärderas med BLEU, COMET, human eval; utmaningar: low-resource, idioms, dokument-level consistency.
**Neural Machine Translation** — Seq2seq encoder-decoder (transformer) som mappar källspråk till målspråk end-to-end. Attention ersatte fixed bottleneck; multilingual models (mBART, NLLB) delar parametrar över språkpar.
**Back-Translation** — Översätt monolingual target data till source med MT system, träna på synthetic parallel corpus. Kraftfull data augmentation för low-resource MT; kvalitet begränsad av back-translation model errors.
**Text Summarization** — Komprimera dokument till kort sammanfattning som bevarar nyckelinformation. Extractive (select sentences) vs abstractive (generate new phrasing); long-document summarization utmanande för context limits.
**Extractive Summarization** — Välj viktiga meningar från källtext via scoring (TF-IDF, TextRank, neural sentence ranker). Faithful (ingen hallucination) men begränsad omformulering; baseline för nyhetsartiklar och legal text.
**Abstractive Summarization** — Generera ny formulering som sammanfattar innehåll; kan parafrasera och synthesize. Seq2seq, BART, GPT; risk för hallucination; ROUGE eval mot reference summaries.
**Question Answering** — Svara på frågor givet kontext (reading comprehension) eller öppen kunskap (retrieval + generation). Span extraction (SQuAD), generative QA, multi-hop reasoning; LLMs kombinerar parametric och retrieved knowledge.
**Reading Comprehension** — Svara baserat på given passage; SQuAD-style span extraction eller generative answer. Tests understanding och grounding; benchmark för encoder och encoder-decoder models före LLM era.
**Open-Domain QA** — Hämta relevant dokument från corpus/web + generera eller extrahera svar. Pipeline: retriever → reader; RAG för LLMs; utmaningar: retrieval quality, attribution, multi-hop.
**Closed-Book QA** — Svara enbart från parametrisk modellkunskap utan external retrieval. TriviaQA, Natural Questions closed-book eval; testar memorization och reasoning in weights; hallucination risk vid saknad kunskap.
**Sentiment Analysis** — Klassificera positiv/negativ/neutral attityd i text (reviews, social media). Lexicon-based → neural → LLM; domain shift (sarcasm, negation) utmanande; multilingual sentiment kräver culture-aware data.
**Aspect-Based Sentiment** — Sentiment per aspekt (mat, service, pris) i recensioner snarare än document-level. Fine-grained opinion mining; sequence labeling eller structured extraction; viktigt för product analytics.
**Text Classification** — Tilldela dokument en eller flera kategorier (topic, intent, spam). BOW → CNN/RNN → BERT fine-tune → LLM zero-shot; class imbalance och label noise vanliga problem.
**Multi-Label Classification** — Flera etiketter samtidigt per dokument (t.ex. taggar, ICD codes). Binary relevance, classifier chains, or multi-label softmax; eval med micro/macro F1; label correlation kan exploateras.
**Sequence Labeling** — Per-token labels: NER, POS, chunking. BiLSTM-CRF historiskt; transformer token classification head; CRF layer kan förbättra consistency; alignment issues med subword tokenization.
**Sequence-to-Sequence** — Mappa input-sekvens till output-sekvens: MT, summarization, data-to-text. Encoder-decoder med cross-attention; autoregressive decoding; exposure bias mitigated via scheduled sampling, RL.
**Encoder-Decoder for NLP** — Transformer encoder-decoder (T5, BART, mT5) för conditional generation tasks. Encoder ser full input bidirectionally; decoder autoregressiv med cross-attention; unified text-to-text framework.
**Masked Language Model** — MLM: predice maskerade tokens från omgivande kontext; BERT pretraining objective. Bidirectional context; [MASK] token under träning; MLM + NSP (deprecated) → modern variants utan NSP.
**Causal Language Model** — Autoregressiv nästa-token prediction P(x_t|x_{<t}); GPT-style left-to-right. Enables generation and in-context learning; no bidirectional context during pretrain; standard for decoder-only LLMs.
**Prefix LM** — Bidirectional attention på prefix + causal på suffix; GLM-style. Kombinerar understanding och generation; används i vissa Chinese LLMs och unified pretraining objectives.
**UL2 Objective** — Mixture of denoisers med varied mask patterns (R, S, X modes) i unified framework. Google T5 v2 approach; single model multi-task via different corruption types; improves transfer across task types.
**Span Corruption** — Maskera contiguous token spans och träna modell att fylla i dem; T5 "text-to-text" pretraining objective. Generaliserar MLM till multi-token prediction; effektiv för encoder-decoder architectures.
**Text Infilling** — Fyll i luckor/gaps i text som denoising objective; relaterat till span corruption och BART. Flexibel corruption pattern; används i infilling-capable LLMs (Codex, GPT infilling modes).
**Denoising Autoencoder for Text** — Rekonstruera korrupt input (noise, deletion, permutation) till original; BART-style. Combines bidirectional encoder med autoregressive decoder; stark för summarization och generation.
**Word Embeddings** — Dense lågdimensionella vektorer för ord som fångar semantisk och syntaktisk likhet. Word2Vec, GloVe, fastText; statiska (ett vektor per ord); ersatta av contextual embeddings i modern NLP.
**Word2Vec Skip-gram** — Predicera context words från center word (eller vice versa CBOW); shallow neural LM. Negative sampling för effektivitet; berömda analogies (king - man + woman ≈ queen); statiska embeddings.
**GloVe Embeddings** — Global co-occurrence statistics + weighted least squares på log co-occurrence matrix. Kombinerar global matrix factorization med local context windows; populär pre-BERT baseline.
**FastText** — Subword-aware embeddings (character n-grams); robust för morphology och OOV via sum of n-gram vectors. Facebook; stark för morphologically rich languages; används i classification baselines.
**Contextual Embeddings** — Token representation beror på hela meningen/kontexten (ELMo, BERT, GPT hidden states). Samma ord får olika vektorer i olika contexts; löser polysemy problem för statiska embeddings.
**ELMo** — Embeddings from Language Models: deep bidirectional LSTM LM; stack LSTM layers som features. Contextual men pre-transformer; weighted sum of layer representations; införde contextual embeddings brett.
**Subword Regularization** — Sample olika BPE segmenteringar under träning (multiple splits per word). SentencePiece technique; improves robustness och reduces overfitting to specific segmentation; better rare word handling.
**Stemming and Lemmatization** — Reducera ord till stam (stemming, heuristic) eller lemma (lemmatization, morphological analysis). IR och klassiska NLP pipelines; mindre viktigt med subword tokenizers men kvar i search indexing.
**Stop Word Removal** — Ta bort frekventa funktionsord (the, is, at); traditionell IR preprocessing. Mindre viktigt med transformers som lär funktion ordens roll; kan skada för vissa tasks (negation, idioms).
**Bag of Words** — Representera dokument som ordfrekvensvektor utan ordning; enkel baseline. Hög dimensionalitet; ignorera syntax; fortfarande baseline för text classification och topic modeling (LDA).
**TF-IDF Vectorization** — Term frequency × inverse document frequency; sparse representation downweighting common terms. Standard för IR, document classification baselines; kombineras med linear SVM/logistic regression.
**N-gram Language Model** — Predicera P(w_t|w_{t-n+1},...,w_{t-1}) från n-1 föregångare; Markov assumption. Smoothing (Kneser-Ney) för zero counts; ersatt av neural LMs men kvar i speech och baselines.
**Perplexity in NLP** — exp(cross-entropy loss) per token; standard LM evaluation, lägre = bättre. Comparable across models on same vocab/test set; correlates med downstream sometimes but not always.
**Kneser-Ney Smoothing** — Avancerad smoothing för n-gram LM som använder lower-order distributions för unseen n-grams. State-of-art count-based LM smoothing; backoff/interpolation; historisk MT och speech baseline.
**Vocabulary Coverage** — Andel text som kan tokeniseras utan UNK/byte fallback; viktigt för multilingual models. Low coverage ⇒ information loss; byte-level tokenizers achieve ~100% coverage.
**Out-of-Vocabulary Handling** — Strategier för okända ord: UNK token, byte fallback, subword decomposition, character models. Modern subword/byte tokenizers eliminerar UNK; OOV nu mer "rare token fragmentation" problem.
**Morphological Analysis** — Analysera morfem, stam, böjningsformer (plural, case, tense). Viktigt för agglutinative languages (Finnish, Turkish); finite-state tools; neural models often learn morphology implicitly.
**Multilingual NLP** — Modeller och pipelines för flera språk: shared vocab, language-specific adapters, cross-lingual transfer. mBERT, XLM-R, NLLB; challenges: script diversity, resource imbalance, code-switching.
**Cross-Lingual Transfer** — Träna på source language(s), applicera på target utan eller med minimal target data. Shared multilingual pretraining enables zero-shot transfer; typologisk närhet och shared vocab påverkar transfer quality.
**Zero-Shot Cross-Lingual** — Ingen target language labeled data vid träning; model generalizes via multilingual pretrain. XNLI, XTREME benchmarks; quality varies greatly by language pair and resource level.
**Language Identification** — Klassificera vilket språk text tillhör; preprocessing för multilingual pipelines. fastText lid, CLD; viktigt före routing till rätt model eller tokenizer; code-switching complicates.
**Code-Switching** — Blandning av språk inom samma yttrande eller konversation. Common in social media; challenges tokenization, POS, MT; multilingual models partially handle; dedicated CS corpora scarce.
**Transliteration** — Konvertera text mellan skript (latin ↔ Cyrillic, romanization of Arabic/Chinese). Useful for search, MT preprocessing; rule-based and neural; not same as translation (same language, different script).
**Text Normalization** — Expandera contractions (don't → do not), standardisera unicode (NFKC), lowercase, handle emojis. Preprocessing för ASR output, social media NLP; LLMs increasingly robust without heavy normalization.
**Truecasing** — Återställ korrekt versalisering (proper nouns, sentence start) i lowercased ASR/NLP output. Sequence labeling eller language model; viktigt för display och downstream NER.
**Sentence Segmentation** — Dela text i meningar (sentence boundary detection); punkt disambiguation (Dr. vs period). Rule-based, statistical, neural; prerequisite för many NLP pipelines; tricky for social text.
**Word Alignment** — Mappa ord mellan käll- och målmening i parallel corpus; MT training och analysis. IBM models, GIZA++; neural attention partially replaces explicit alignment; useful for error analysis.
**BLEU for MT** — N-gram precision mot reference translation(s) med brevity penalty; de facto MT metric. Correlates imperfectly with human judgment; gaming via short outputs; COMET/neural metrics preferred now.
**COMET Metric** — Neural MT evaluation metric tränad att korrelera med human quality judgments. Uses cross-lingual embeddings; segment and reference-based; WMT metrics shared task winner; better than BLEU for modern NMT.
**Dialogue Systems** — Konversationsagent som hanterar multi-turn context, user goals och system responses. Task-oriented (slot-filling) vs open-domain (chitchat); LLM chatbots unify both med prompting och tools.
**Task-Oriented Dialogue** — Slot-filling för specifika mål: booking, support, information lookup. Pipeline: NLU → DST → policy → NLG; or end-to-end neural; benchmarks: MultiWOZ, Schema-Guided Dialogue.
**Open-Domain Dialogue** — Chitchat utan specifikt task goal; fokus på engagement, coherence, persona. PersonaChat, DailyDialog; LLMs (ChatGPT) dominerar; utmaningar: repetition, consistency, safety.
**Response Generation** — Generera relevant, coherent svar givet dialoghistorik och ev. knowledge. Seq2seq, retrieval-augmented, LLM prompting; eval: human rating, perplexity, diversity metrics (distinct-n).
**Dialogue State Tracking** — Spåra user intent, slots och constraints över turns i task-oriented dialogue. Set-based eller span-based DST; critical för multi-turn booking; LLMs kan implicit track via context window.
**Intent Classification** — Klassificera användarens avsikt (book_flight, check_balance) från utterance. NLU component; text classification; confusion mellan similar intents; LLM zero-shot via description.
**Slot Filling** — Extrahera parametrar (datum, plats, tid) från user input som slot-value pairs. Sequence labeling eller span extraction; joint intent+slot models; LLM structured output (JSON mode).
**Information Extraction** — Strukturerad data från ostrukturerad text: entities, relations, events, tables. Pipeline för knowledge base population; LLMs med prompting och fine-tuning; eval on ACE, TAC-KBP.
**Relation Extraction** — Identifiera relationer mellan entiteter (employed_by, located_in). Supervised on sentence or document level; distant supervision från KB; LLM relation extraction via prompts.
**Event Extraction** — Hitta händelser (trigger + arguments: agent, theme, time, place). ACE event schema; complex structured prediction; viktigt för news, intelligence, clinical NLP.
**Template Filling** — Populera fördefinierat schema/slots från text (product specs, job postings). Structured prediction; overlaps med slot filling; LLM JSON extraction enables flexible templates.
**Semantic Parsing** — Mappa naturligt språk till logisk form, SQL, SPARQL eller executable program. Compositional semantics; GeoQuery, ATIS benchmarks; LLM text-to-SQL revolutionerar men hallucination kvarstår.
**Text-to-SQL** — Generera SQL-fråga från naturligt språk question givet databasschema. Spider, BIRD benchmarks; schema linking och execution accuracy eval; enterprise copilots (Copilot, Text2SQL tools).
**SPARQL Generation** — Generera SPARQL queries mot RDF knowledge graphs från naturligt språk. Semantic Web applications; less mainstream than SQL men viktigt för linked data; LLM + ontology constraints.
**Constituency-to-Logic** — Kompositionell semantics: syntax-driven mapping till formella meaning representations (lambda calculus, DRT). Montague semantics tradition; neural semantic parsing approximates.
**Discourse Parsing** — Analysera struktur och relationer över meningar i dokument (RST, PDTB). Rhetorical Structure Theory trees; important for summarization, coherence; long-context LLMs partially capture.
**Rhetorical Structure Theory** — RST: nucleus-satellite relationer (elaboration, contrast, cause) i dokumentträd. Discourse analysis framework; RST parsers och eval corpora; informs document-level generation.
**Coherence Modeling** — Bedöm om text hänger ihop logiskt: entity continuity, discourse relations, topic flow. Entity grid models, neural coherence scoring; important for generation eval och essay scoring.
**Lexical Substitution** — Hitta alternativa ord som passar samma kontext (SemEval LexSub task). WordNet-based och contextual (BERT masked prediction); används i paraphrase, data augmentation, writing assistance.
**Paraphrase Generation** — Omformulera mening med bibehållen betydelse; data augmentation och simplification. Seq2seq, back-translation, LLM prompting; eval med BLEU, BERTScore, human judgment.
**Paraphrase Detection** — Avgör om två meningar/text är parafraser (entailment-like). Quora, MRPC benchmarks; viktigt för deduplication, plagiarism detection, QA training data filtering.
**Semantic Textual Similarity** — STS: score grad av meningsslikhet (0-5) snarare än binary paraphrase. SemEval STS tracks; embedding cosine, cross-encoder, LLM scoring; used in retrieval and clustering.
**Natural Language Inference** — NLI: klassificera relation entre premiss och hypotes som entailment, contradiction, neutral. SNLI, MultiNLI benchmarks; fundamental language understanding; transfer to many tasks.
**Recognizing Textual Entailment** — RTE: klassisk NLI-uppgift från PASCAL challenges; föregångare till modern NLI. Directional inference: does text A entail text B? Foundation for zero-shot classification via hypothesis templates.
**Stance Detection** — Klassificera pro/con/neutral attityd mot specifikt target (person, policy) i text. SemEval stance tasks; related to sentiment but target-specific; important for social media analysis and debate mining.
## Optimering & Träning

**Loss Function** — Skalär målfunktion L(θ) som minimeras under träning; mäter avstånd mellan prediktion och target. Val (CE, MSE, hinge, contrastive) styr vad modellen optimerar; olika losses inducerar olika decision boundaries och robusthet.
**Cross-Entropy Loss** — För klassificering: -Σ y log ŷ; ekvivalent med negativ log-likelihood vid one-hot targets. Standard för multi-class (med softmax) och language modeling (per-token); numeriskt stabil med log-softmax.
**Mean Squared Error** — MSE = (1/n)Σ(y-ŷ)²; standard regression loss som straffar stora fel kvadratiskt. Känslig för outliers; antar Gaussian noise model; används i autoencoders, diffusion noise prediction, value heads.
**Huber Loss** — Robust regression: kvadratisk nära noll, linear för stora fel (δ-tröskel). Mindre känslig för outliers än MSE; smooth transition; används i object detection (Smooth L1) och robust regression.
**Hinge Loss** — SVM loss max(0, 1 - y·ŷ) för margin-maximering; encouragerar korrekt klassificering med margin. Convex; används i max-margin classifiers och some ranking losses; zero gradient vid satisfied margin.
**Focal Loss for Classification** — Down-weight lätta/well-classified exempel: -(1-p_t)^γ log(p_t); fokuserar träning på hard examples. RetinaNet för object detection; adresserar class imbalance och easy negative dominance.
**Label Smoothing** — Mjuka targets (1-ε, ε/K) istället för hard one-hot; motverkar overconfidence och kalibrering problem. ε typiskt 0.1; förbättrar generalization och model calibration; standard i transformer träning.
**Class Weights in Loss** — Vikta sällsynta klasser högre i CE: w_c proportional to inverse frequency. Hanterar class imbalance; risk för over-predicting rare classes om weights too aggressive; focal loss alternativ.
**Contrastive Loss** — Dra positiva par nära, push negativa bort i embedding space (InfoNCE, triplet, NT-Xent). Self-supervised och metric learning foundation; temperature τ skalar softmax over similarities.
**Triplet Loss** — max(0, d(a,p) - d(a,n) + margin): anchor närmare positive än negative med margin. Face recognition (FaceNet); kräver hard negative mining; sensitive to margin och distance metric val.
**InfoNCE Loss** — Noise contrastive estimation: -log(exp(sim(q,k+)/τ) / Σ exp(sim(q,k_i)/τ)); lower bound on mutual information. SimCLR, MoCo foundation; batch size och negative count critical.
**NT-Xent Loss** — Normalized Temperature-scaled Cross Entropy i SimCLR; symmetrized InfoNCE over augmented pairs. Normalization (cosine similarity) och temperature critical for performance; large batch helps.
**Denoising Loss** — Rekonstruktionsfel mellan original och denoised/reconstructed input (MSE, CE on masked tokens). BERT MLM, diffusion ε-prediction, autoencoders; corruption strategy defines learning signal.
**Perplexity as Training Metric** — Monitorera exp(average CE) under LM träning; lägre perplexity ⇒ bättre next-token prediction. Not identical to downstream task perf; used for early stopping and comparing LMs on same vocab.
**Gradient Descent Variants** — SGD, momentum, Nesterov, Adam, AdamW, Adafactor m.fl. med olika adaptivity och minneskrav. Val påverkar konvergenshastighet, generalization och stabilitet; ingen universal winner.
**Mini-Batch SGD** — Stochastic gradient på minibatch (typiskt 32-4096+) per steg; balanserar noise och throughput. GPU-effektiv; batch size påverkar implicit regularization och kräver LR scaling.
**Full-Batch Gradient Descent** — Gradient över hela dataset per steg; deterministisk men dyrt för stora data. Sällan i deep learning utom små dataset; används teoretiskt som referens för SGD convergence.
**Learning Rate Finder** — LR range test (Leslie Smith): öka LR exponentiellt per batch, plot loss vs LR, välj LR precis före divergence. Hittar usable LR range snabbt; warmup start point guidance.
**One-Cycle Learning Rate** — Öka LR till max, sedan minska till minimum i ett träningspass (super-convergence). Fast.ai; möjliggör högre peak LR med momentum cycling; accelererar träning.
**Cyclical Learning Rate** — Periodisk variation av LR mellan bounds (triangular, cosine); kan escape sharp minima. Snapshot ensembling vid cycle minima; less common nu med cosine decay + warmup.
**ReduceLROnPlateau** — Sänk LR när valideringsmetrik platear (factor ×0.1 efter patience epochs). Adaptive till dataset; kräver reliable validation signal; används när fixed schedule okänd.
**Step Decay Schedule** — Multiplicera LR med faktor (t.ex. 0.1) var N:e epoch eller vid fixed milestones. Enkel och effektiv; ResNet ImageNet standard; abrupt drops kan destabilize transformers utan warmup.
**Linear Warmup Decay** — Linear warmup från 0 till peak LR, sedan linear decay till 0 över total steps. Common transformer schedule; warmup prevents early instability; decay ensures fine convergence.
**Adam Optimizer** — Adaptive moment estimation: per-parameter LR från first moment m och second moment v med bias correction. β1≈0.9, β2≈0.999; bra default men kan generalize sämre än SGD+mom i vissa CV tasks.
**AdamW Optimizer** — Decoupled weight decay: L2 regularization appliceras direkt på weights, inte via gradient i Adam. Fixar L2+Adam interaction; standard för transformer träning; weight decay typiskt 0.01-0.1.
**Adam Beta Parameters** — β1 (first moment decay, ~0.9) och β2 (second moment, ~0.95-0.999); högre β2 ⇒ längre memory av gradient magnitudes. LLM träning ofta β2=0.95; påverkar adaptivity och early training dynamics.
**Epsilon in Adam** — Liten konstant (~1e-8) i denominator √(v)+ε för numerisk stabilisering vid v≈0. För LLM kan större ε (1e-6) ibland hjälpa; förhindrar division by zero och exploding updates.
**SGD with Momentum** — v_t = μv_{t-1} + g_t; θ -= ηv_t; accumulerar gradient history för snabbare convergence. μ typiskt 0.9; fortfarande competitive i CV med proper schedule; Nesterov variant lookahead.
**Nesterov Accelerated Gradient** — Momentum med lookahead: evaluate gradient at θ + μv; snabbare convergence än vanilla momentum. Geometric intuition: correct overshoot; used in some CV training recipes.
**Adafactor Optimizer** — Memory-efficient Adam variant: factorized second moment för matrices, optional parameter scaling. T5 default; reducerar optimizer state memory för stora modeller; suitable for Adafactor+manual LR schedule.
**8-bit Adam** — Kvantiserade optimizer states (m, v) till 8-bit med block-wise scaling; bitsandbytes. ~75% optimizer memory savings; minimal quality loss för LLM fine-tuning; critical for consumer GPU fine-tuning.
**Sophia Optimizer** — Second-order inspired med diagonal Hessian estimates för curvature-aware updates. Claims 2× faster than Adam on some LLM pretrain; newer, less widespread adoption; uses Hessian diagonal via Hutchinson.
**Muon Optimizer** — Orthogonalized updates för 2D weight matrices via Newton-Schulz iteration; spectral norm control. Experimental; aims for better conditioning of hidden layer updates; used in some scaling research.
**Gradient Accumulation Steps** — Accumulera gradients över N micro-batches före optimizer step; effective batch = micro_batch × N × num_gpus. Simulerar större batch utan proportional memory; sync gradients var N:te step.
**Global Batch Size** — Total antal exempel per optimizer step över alla enheter och accumulation: micro × accum × world_size. Skalning påverkar LR (linear scaling rule), training dynamics och wall-clock.
**Micro Batch Size** — Per-GPU/per-device batch per forward-backward pass; begränsad av GPU memory. Flash attention, gradient checkpointing ökar feasible micro batch; tradeoff mot accumulation steps.
**Gradient Clipping Norm** — Clip global gradient norm ||g|| till max_norm om den överskrider tröskel: g ← g × max_norm/||g||. Förhindrar exploding gradients i RNN/transformer early training; typical max_norm 1.0.
**Gradient Clipping Value** — Element-wise clip av gradient till [-v, v]; mer aggressiv än norm clipping. Mindre vanligt i transformers; används i some RL och RNN training.
**Weight Decay Regularization** — L2 penalty λ||θ||² added to loss eller decoupled direct shrinkage (AdamW); discourages large weights. Typiskt 0.01-0.1 för transformers; improves generalization; not same as L2 in Adam without decoupling.
**L1 Regularization** — λΣ|θ| penalty som inducerar sparsity (many zero weights). Lasso; feature selection i linear models; less common in deep nets except pruning research och sparse coding.
**Dropout Regularization** — Stochastic zeroing av activations (rate p) under träning; ensemble effect. Standard i transformers (attention + FFN dropout); disabled at inference; rate typiskt 0.1-0.3.
**Early Stopping** — Stoppa träning när validation loss/metric inte förbättras efter patience epochs. Förhindrar overfitting; saves compute; requires reliable validation set; model selection bias om HP tuned on same val.
**Patience in Early Stopping** — Antal epochs/steps att vänta efter senaste val improvement före stop. För låg ⇒ premature stop; för hög ⇒ wasted compute on overfitting; typiskt 3-10 epochs.
**Checkpoint Averaging** — Medelvärde av weights från sista k checkpoints eller SWA för stabilare model. Reducerar variance i SGD trajectory; often improves generalization slightly; used in competition-winning models.
**Stochastic Weight Averaging** — SWA: genomsnitt av weights längs training trajectory (ofta med cyclical/high LR phase). Flat minima bias; improves generalization; SWAG extends with Gaussian uncertainty.
**Exponential Moving Average** — EMA: θ_ema ← β θ_ema + (1-β) θ med β≈0.999; smoother weight track för eval/inference. Stable diffusion, some LLM training; EMA model often better calibrated than training weights.
**Teacher EMA** — EMA av teacher weights i self-supervised (MoCo, BYOL, Mean Teacher); student updated via gradient, teacher via EMA. Consistent target för contrastive/distillation; prevents collapse i BYOL.
**Batch Normalization Momentum** — Running mean/var uppdateras: stat ← momentum×stat + (1-momentum)×batch_stat; typiskt 0.1-0.99. Inference uses running stats; momentum controls adaptation speed to distribution shift.
**SyncBatchNorm** — Synkroniserar batch statistics över GPUs under distributed träning; effective batch size för BN stats. Necessary för small per-GPU batch i detection/segmentation; increases communication overhead.
**Loss Scaling in FP16** — Multiplicera loss med scale factor S före backward för att förhindra gradient underflow i FP16. Gradients unscaled före optimizer step; static eller dynamic scaling.
**Dynamic Loss Scaling** — Automatisk justering av FP16 scale: öka om no overflow, halvera vid Inf/NaN gradient. Mixed precision training standard; AmpGrad/Apex/PyTorch AMP implement.
**Gradient Overflow Detection** — Skip optimizer step om Inf/NaN i gradient; halvera loss scale och retry. Critical för stable FP16 training; logged as overflow events; may indicate LR too high.
**Automatic Mixed Precision** — AMP: forward/backward i FP16/BF16, master weights och optimizer i FP32. torch.cuda.amp.autocast + GradScaler; ~2× speedup på tensor cores; standard training practice.
**BFloat16 Training** — BF16 mixed precision: samma exponent range som FP32 (8 bit), mindre mantissa (7 bit). Prefererad för LLM träning (A100+); less loss scaling needed vs FP16; slightly lower precision acceptable.
**TF32 on Ampere** — TensorFloat-32: 19-bit matmul på NVIDIA Ampere+ (default i cuDNN/PyTorch). ~FP32 range med reduced precision; transparent speedup; minimal accuracy impact för de flesta DL workloads.
**Distributed Data Parallel** — DDP: replikerad modell per GPU, gradient all-reduce efter backward, synkroniserat optimizer step. PyTorch standard; effektiv för multi-GPU; bucketed all-reduce för overlap.
**Horovod** — Uber's distributed training framework med ring allreduce för gradient synchronization. Framework-agnostic (TF, PyTorch); alternative to DDP; efficient at scale med hierarchical allreduce.
**DeepSpeed ZeRO Stage 3** — Shardar parametrar, gradients och optimizer states över GPUs; gather on demand per layer. Enables träning av modeller större än single GPU memory; communication overhead vs memory savings.
**Optimizer State Sharding** — ZeRO-1/2: dela Adam m,v states (och gradients) över enheter; each GPU stores fraction. Reducerar optimizer memory ~Nx; prerequisite för large model fine-tuning utan full ZeRO-3.
**Pipeline Parallel Schedule** — Dela modell i stages på olika enheter; 1F1B schedule minimerar pipeline bubble (idle time). GPipe, PipeDream; för very large models kombinerat med tensor parallel.
**Tensor Parallel All-Reduce** — Efter column-parallel matmul: all-reduce partial results; row-parallel följer med split. Megatron-LM style; splits attention/FFN across GPUs; communication per layer.
**Communication Overlap** — Överlappa gradient all-reduce med backward compute (backward hook scheduling). Reducerar distributed training overhead; DeepSpeed/PyTorch DDP bucket optimization; critical at scale.
**Training Instability** — Loss spikes, NaN/Inf, divergence; orsaker: LR too high, bad init, precision issues, bad data batch. Mitigation: LR reduction, gradient clip, loss scale adjust, checkpoint rollback.
**Loss Spike Recovery** — Rollback till checkpoint före spike, sänk LR (÷2-10), optionally skip offending batch. GPT-3 training recipe; some frameworks auto-detect och recover; log spike frequency.
**NaN Detection Hook** — Abort eller skip step vid NaN i loss eller gradient; alert operators. torch.autograd anomaly detection for debug; production training monitors loss finiteness.
**Weight Initialization Scale** — För stor init ⇒ activation/gradient explosion; för liten ⇒ vanishing signal. Xavier/He scaling; residual scaling (1/√N); depth-dependent init critical för deep nets.
**Residual Scaling Init** — Skala residual branch init med 1/√(2×num_layers) för stabil djup träning. GPT, Post-LN transformers; prevents growth of activation variance through layers.
**Learning Rate vs Batch Size Scaling** — Linear scaling rule: dubbla batch ⇒ dubbla LR (till breakpoint); motivated av gradient noise scale. Goyal et al.; breaks vid very large batch; warmup compensates.
**Square Root Scaling Rule** — Alternativ: η ∝ √(batch); less aggressive LR increase med batch size. Some empirical success when linear scaling diverges; middle ground between constant LR and linear.
**Warmup Steps Calculation** — Typiskt 1-5% av total training steps ( eller fixed 2000-10000); linear warmup till peak LR. Stabiliserar Adam early phase; longer warmup for larger batch/model.
**Total Training Tokens** — Budget i tokens (billions/trillions) för LLM pretrain; primary scaling axis med params. Chinchilla optimal: balance tokens och params; undertrain = waste compute on params.
**Chinchilla Token Budget** — ~20 tokens per parameter för compute-optimal pretrain (Hoffmann et al.). 70B model ⇒ ~1.4T tokens; prior Kaplan undertrained models; reshaped industry compute allocation.
**Over-Training** — Träna längre än Chinchilla-optimal (mer tokens per param); worse compute-efficiency men bättre inference-time performance per param. Llama 3 etc. overtrain for deployment efficiency.
**Data Mixture Ratio** — Viktning av subcorpora (web, code, books, wiki) i pretrain blend; strongly affects capabilities. Code-heavy ⇒ coding; deduplication och quality filtering per source; tuned empirically.
**Curriculum Sampling** — Öka andel svårare/längre data under träning; progressive difficulty. Sequence length curriculum common; quality curriculum less standardized; can stabilize early training.
**Token-Level Loss Masking** — Exkludera padding tokens och vissa tokens (user vs assistant) från loss computation. Cross-entropy masked where attention_mask=0; instruction tuning masks prompt tokens.
**Sequence Length Curriculum** — Börja med kort max sequence (2K), öka gradvis till target (8K, 32K). Reducerar early training cost; RoPE/YaRN extrapolation for extension; memory scales quadratically without FlashAttention.
**Flash Attention Training** — IO-aware exact attention algorithm; O(N) memory vs O(N²); enables longer sequences. Integrated in PyTorch 2.0+, HuggingFace; prerequisite for long-context LLM training.
**Fused Optimizer Kernels** — Combined Adam update kernels (fused Adam) minskar kernel launch overhead. Apex, DeepSpeed, PyTorch fused optim; modest speedup; important at scale with many small tensors.
**torch.compile Training** — PyTorch 2 graph compilation (Inductor) accelererar forward/backward via fusion och optimization. mode="reduce-overhead" or "max-autotune"; dynamic shapes tricky for variable seq len.
**Profiling Training Step** — PyTorch profiler, Nsight Systems för CPU/GPU bottleneck analysis. Identifies dataloader stalls, kernel gaps, excessive all-reduce; essential for scaling efficiency (MFU improvement).
**MFU Measurement** — Model FLOPs Utilization: achieved FLOPs / peak hardware FLOPs; efficiency metric for LLM training. Typical 30-55% on large clusters; accounts for communication, recomputation, dataloader.
**Throughput Tokens per Second** — Träningshastighet i tokens/s (global, all GPUs); primary operational metric. Function of MFU, batch size, seq len, hardware; used for time-to-train estimates och cost projection.
**Time to Train Estimate** — Wall-clock ≈ total_tokens / (throughput × num_nodes efficiency); projicerar completion. Includes checkpointing overhead, failures, maintenance; critical for cluster scheduling och budget.
**Hyperparameter Sensitivity** — Robusthet mot små HP-ändringar (LR ±20%, batch ±2×); fragile methods unreliable. Good methods show stable performance across reasonable HP range; report sensitivity i papers.
**Seed Variance** — Resultatvariation över random seeds (init, dropout, data order); can be ±0.5-2% on metrics. Report mean±std over 3-5 seeds for reliable claims; some benchmarks highly seed-sensitive.
**Deterministic Training** — torch.use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG, fixed seeds för reproducerbarhet. Slower (no benchmark mode); slight numerical differences eliminated; required for debugging och regression tests.
**Non-Deterministic Training** — Default: atomicAdd order, cudnn benchmark, flash attention non-determinism. Faster; acceptable slight run-to-run variation; report average over seeds for eval.
**Cudnn Benchmark Mode** — torch.backends.cudnn.benchmark=True autotune conv algorithms för fastest; non-deterministic. Standard for production CV training; disable for reproducibility experiments.
**DataLoader Bottleneck Fix** — Öka num_workers, prefetch_factor, pin_memory=True, persistent_workers; optimize preprocessing. GPU idle waiting for CPU data common bottleneck; NVIDIA DALI for GPU decoding; cache preprocessed data.
## Generativ AI & GANs

**Generative AI** — AI som skapar nytt innehåll som text, bild, ljud, video och kod. Omfattar LLM:er, diffusion, GAN:er och VAE:er.
**Generative Model** — Modellerar datadistribution P(x) eller villkorad P(x|y) för sampling av realistiska utdata. Tränas via MLE, adversarial-, diffusions- eller flow-mål.
**Discriminative Model** — Modellerar P(y|x) direkt för klassificering och regression utan att modellera indatafördelningen. Kan inte generera samples men används i generativa pipeliner.
**Generative Adversarial Network** — GAN: generator G och diskriminator D i minimax-spel där G försöker lura D. Implicit densitetsmodell med instabilitet och mode collapse.
**Generator Network** — G(z) mappar latent brus z till syntetiska samples i måldomänen. Arkitekturen varierar från DCGAN till StyleGAN mapping network.
**Discriminator Network** — D(x) skiljer äkta från falska samples och ger lärgradient till G. Spektralnorm och gradientstraff stabiliserar träningen.
**Minimax GAN Objective** — min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]. Nash-jämvikt vid p_G = p_data; gradient försvinner om D blir för stark.
**Non-Saturating GAN Loss** — Generatorförlust -log D(G(z)) ger starkare gradient än log(1-D(G(z))) när D(G(z))≈0. Standardmål för G-träning.
**Mode Collapse in GAN** — Generatorn producerar begränsad variation trots divers latent z. D kan inte ge gradient för saknade modeller i datadistributionen.
**Training Instability in GAN** — Oscillerande förluster och dominans av D eller G gör träningen instabil. WGAN-GP och spektralnorm förbättrar konvergens.
**Vanishing Gradient in GAN** — När D blir för stark får G ingen användbar gradient eftersom förlusten mättas ut. Icke-mättnad förlust mildrar problemet.
**Wasserstein GAN** — WGAN minimerar Wasserstein-1-avstånd via dualitet och kräver 1-Lipschitz-kritiker. Viktclippning eller gradientstraff ger stabilare träning.
**WGAN-GP** — Gradientstraff λE[(||∇D(x̂)||₂-1)²] ersätter viktclippning för Lipschitz-tvång. Standard-WGAN-variant med λ typiskt 10.
**Spectral Normalization GAN** — Normaliserar D:s viktmatriser via spektralnorm så Lipschitz-konstanten ≤ 1. Enklare än gradientstraff; används i SNGAN och BigGAN.
**Progressive GAN** — Upplösningen ökas gradvis under träning med fade-in av nya lager. Stabiliserar högupplöst GAN-träning och fotorealistiska ansikten.
**StyleGAN** — Stylebaserad generator med mapping network z→w och AdaIN per lager. Oövervakad stilkontroll av pose, identitet och bakgrund.
**StyleGAN2** — Förbättrad arkitektur med weight demodulation och path length regularization. Högre kvalitet och färre artefakter än StyleGAN.
**StyleGAN3** — Alias-fri generator med bättre ekvivarians vid translation och rotation. Minskar texture sticking och förbättrar animation.
**BigGAN** — Storskalig klassvillkorad GAN med BigBatch, spektralnorm och self-attention. State-of-the-art på ImageNet men kräver stor beräkning.
**Conditional GAN** — cGAN villkorar G och D på label y via konkatenering eller projektion. Möjliggör kontrollerad generering och text-till-bild.
**AC-GAN** — Auxiliär klassificeringsförlust på D utöver äkta/falsk-diskriminering. Stabiliserar villkorad träning och förbättrar klasskvalitet.
**CycleGAN** — Obepaarat bild-till-bild via cykelkonsistens G_AB(G_BA(x))≈x utan parvisa data. Används för stilöverföring som häst↔zebra.
**Pix2Pix** — Bepaarat bild-till-bild med villkorad GAN och L1-rekonstruktionsförlust. Kräver justerade par; L1 förhindrar felaktiga utdata.
**SRGAN** — Super-upplösnings-GAN med perceptuell VGG-förlust och adversarial förlust. Perceptuellt skarpare än rena MSE-metoder.
**ProGAN** — Progressiv uppväxling med fade-in mellan upplösningar för högkvalitativa ansikten. Milstolpe som influerade StyleGAN-arkitekturen.
**DCGAN** — Djup convolutionell GAN med riktlinjer: strided conv, BatchNorm och försiktig initiering. Etablerade conv-GAN-bästa praxis.
**Latent Space Interpolation** — Linjär interpolation z1→z2 i latent space ger mjuka visuella morfer. Används i ansiktsmorfing och StyleGAN w-mixning.
**Latent Vector z** — Brusvektor z ~ N(0,I) som input till generatorn, typiskt d=512 i StyleGAN. GAN-inversion hittar z som rekonstruerar en given bild.
**GAN Inversion** — Hitta latent kod z eller w så G(z) rekonstruerar given bild via optimering eller encoder. Möjliggör redigering av riktiga bilder.
**Perceptual Loss** — L2 på feature maps från förtränat nätverk (VGG) istället för pixel-L2. Fångar semantisk likhet i SRGAN och style transfer.
**Feature Matching Loss** — Matchar medelvärdet av intermediära D-features mellan äkta och falska batcher. Stabiliserar G-träning och minskar mode collapse.
**Historical Averaging GAN** — Regulariserar G mot löpande medelvärde av tidigare G-parametrar. Minskar oscillation men är mindre vanligt idag.
**Unrolled GAN** — Generatorförlust inkluderar k steg D-optimering unrollade för att förutse D:s svar. Stabilare G-gradienter men dyrare beräkningsmässigt.
**Self-Attention GAN** — SAGAN: self-attention i G och D för långväga beroenden i bilder. BigGAN använder attention för global struktur.
**Projection Discriminator** — Klassembedding projiceras på D:s features via skalarprodukt. Starkare villkorning än konkatenering; används i BigGAN.
**R1 Gradient Penalty** — Regulariserar D med γ/2 E[||∇_x D(x)||²] endast på äkta data. Standard i StyleGAN2 med γ typiskt 10.
**Consistency Regularization GAN** — CR-GAN: konsistensförlust på D för augmenterade omarkerade data. Semi-overvakad träning med få etiketter.
**Energy-Based Model** — EBM tilldelar låg energi till data och hög till andra regioner; P(x) ∝ exp(-E(x)). Kopplar till diffusions- och scoremodeller.
**Normalizing Flow** — Invertibel transform med traktabel Jacobian och exakt log-likelihood via variabelbyte. RealNVP och Glow ger exakt densitet.
**RealNVP** — Affina coupling-lager transformerar hälften av x villkorat på resten med effektiv log-det. Grund för Glow-modellen.
**Glow** — Flow-modell med 1×1 inverterbar conv och affina coupling-lager. Exakt likelihood och effektiv parallell bildgenerering.
**Variational Autoencoder** — VAE: encoder q(z|x), decoder p(x|z), maximerar ELBO. Generativ med latent flaskhals; samples kan bli suddiga.
**Reparameterization Trick** — z = μ + σ⊙ε, ε~N(0,I) möjliggör backprop genom stokastisk sampling. Grundläggande för gradientbaserad VAE-träning.
**KL Divergence in VAE** — KL(q(z|x)||p(z)) regulariserar latent mot prior N(0,I) och motverkar posterior collapse. β-VAE och KL-annealing styr vikten.
**β-VAE** — ELBO med viktad KL β·KL, β>1 för disentanglade representationer. Byter rekonstruktionskvalitet mot faktoriserade latenter.
**Vector Quantized VAE** — VQ-VAE: diskreta latenta koder via vektorquantisering med inlärd codebook. Grund för autoregressiv prior (DALL-E steg 1).
**Codebook in VQ-VAE** — Inlärd embedding-tabell; encoder-utdata quantiseras till närmaste kod. Commitment loss och straight-through-estimator.
**Autoregressive Generative Model** — Faktoriserar P(x) = Π P(x_i|x_{<i}) med sekventiell sampling. Traktabel likelihood men långsam generering.
**PixelCNN** — Autoregressiv bildgenerering pixel för pixel med maskerade conv-lager. Exakt likelihood men mycket långsam sampling.
**WaveNet** — Autoregressiv råvågformsgenerering med dilated causal conv. Högkvalitativ TTS före neural vocoders; långsam inferens.
**Transformer LM as Generator** — GPT-liknande transformer genererar text token för token autoregressivt. Temperatur och top-p styr diversitet.
**Diffusion as Generative Model** — Framåt brusprocess plus inlärnd omvänt denoising utan adversarial träning. State-of-the-art för bild, ljud och video.
**Flow Matching Generative** — Kontinuerliga normaliserande flöden via flow matching utan ODE-simulering under träning. Enklare träning än score matching.
**Consistency Model Generation** — Fåstegsgenerering via consistency distillation från diffusion-lärare. Byter kvalitet mot hastighet för realtidsapplikationer.
**Score-Based Generative Model** — Lär brusvillkorad score function ∇_x log p(x) för sampling via Langevin-dynamik eller SDE. Förenar diffusion och EBM.
**Implicit Generative Model** — Sample utan traktabel densitet, t.ex. GAN via G(z). Utvärderas via FID, IS och mänsklig utvärdering, inte likelihood.
**Explicit Likelihood Model** — Traktabel P(x): autoregressiva modeller, normaliserande flöden och VAE (ELBO). Möjliggör likelihood-baserat modellval.
**Evaluating Generative Models** — FID, IS, precision/recall och mänsklig A/B; inget perfekt enskilt mått. Avvägning mellan kvalitet och diversitet.
**Precision and Recall for GANs** — Precision mäter realistiska samples; recall täcker data-modeller. Detekterar mode collapse via k-NN i feature space.
**Fréchet Inception Distance** — FID: Fréchet-avstånd mellan Gaussiska passningar till Inception-v3-features. Lägre är bättre; standardmått för bildgenerering.
**Inception Score** — IS mäter klassificerbarhet och diversitet hos genererade bilder via Inception-klassificerare. Högre är bättre; largt ersatt av FID.
**Kernel Inception Distance** — KID: MMD med polynomkernel i Inception-features; opartisk estimator jämfört med FID. Bättre för små sample-storlekar.
**Neural Audio Codec** — EnCodec/DAC komprimerar ljud till diskreta tokens vid låg bitrate. Möjliggör ljud-LM som AudioLM och MusicGen.
**Neural Vocoder** — Genererar vågform från mel-spektrogram: WaveGlow, HiFi-GAN, BigVGAN. Bro mellan akustisk modell och hörbar output.
**HiFi-GAN** — Högkvalitativ GAN-vocoder med multi-receptive field fusion och multi-scale D. Standard TTS-vocoder; realtidsbar.
**Mel Spectrogram** — Mel-skalerat log-power-spektrogram; standardfeature för TTS och ASR. Typiskt 80 mel-bins; vocoder rekonstruerar vågform.
**Text-to-Music Generation** — Generera musik från textprompt eller strukturerad spec (genre, tempo). MusicGen och Suno; långformstruktur utmanande.
**Text-to-Video Generation** — Diffusion/transformer genererar videosekvenser från text. Temporal konsistens och fysikplausibilitet är öppna problem.
**Video GAN** — Generera videobilder med temporal konsistens via 3D-conv eller rekurrent tillstånd. Largt ersatt av video-diffusion.
**3D Generative Model** — Generera meshes, NeRF eller 3D Gaussians från text eller bild. Viktigt för spel, AR och robottillämpningar.
**DreamFusion** — Text-till-3D via 2D diffusion-distillation och NeRF-optimering. Renderade vyer matchar diffusion-modellens scores utan 3D-data.
**Generative Fill** — Inpainting/outpainting: fyll maskerade regioner med sammanhängande innehåll. Villkoras på mask och omgivande kontext.
**ControlNet for Generation** — Injicerar spatial kontroll (kanter, djup, pose) i frusen diffusion via sidonätverk. Vanligt i Stable Diffusion.
**LoRA for Style Generation** — Low-rank adaptation för personlig stil utan full omträning. DreamBooth+LoRA för anpassade motiv; effektiv finjustering.
**Prompt Engineering for T2I** — Formulera effektiva textprompter för text-till-bild med beskrivande tokens och stilreferenser. Kvalitet är promptberoende.
**Negative Prompt Engineering** — Specificera oönskade element i negativ prompt för att undertrycka dem via CFG. Minskar vanliga fellägen i Stable Diffusion.
**Seed Control in Generation** — Fixera slumpseed ger reproducerbara utdata givet samma modell och prompt. Viktigt för iterativ förfining.
**Batch Generation** — Generera många samples parallellt med delad modell-forward. Används för dataset och A/B-jämförelse; GPU-minne begränsar batch.
**Classifier Guidance** — Justera diffusion score med ∇_x log p(y|x) från klassificerare för villkorad generering. Ersatt av CFG i praktiken.
**CFG in Diffusion** — Classifier-Free Guidance: guida via (1+w)ε_cond - w·ε_uncond utan separat klassificerare. w styr promptlojalitet vs diversitet.
**Generative Model Safety** — NSFW-filter, vattenmärkning och C2PA-metadata i generativ pipeline. Förhindrar missbruk men filter är ofullkomliga.
**Synthetic Data Generation** — Generera träningsdata med generativa modeller för augmentation och integritet. Kvalitetsfiltrering essentiell; risk för model collapse.
**Data Augmentation via Generation** — GAN/diffusion skapar extra träningsexempel för sällsynta scenarier. Domängap mellan syntetisk och riktig data måste hanteras.
**Privacy-Preserving Generation** — Syntetisk data bevarar statistik utan att memorera PII. Differential privacy och medlemsinferens validerar integritet.
**Membership Inference on Generative Models** — Attack avgör om sample fanns i träningsdata baserat på modellutdata. Integritetsrisk; mildras av differential privacy.
**Model Collapse from Synthetic Data** — Iterativ träning på AI-genererad data degraderar diversitet när svansar försvinner. Blandning av riktig och syntetisk data hjälper.
**Human Evaluation of Generative Output** — A/B-preferenstester, Elo och Likert-skala för perceptuell kvalitet. Guldstandard men dyr och subjektiv.
**Turing Test for Generative AI** — Kan AI-genererat innehåll skiljas från mänskligt av utvärderare? Ofullkomligt benchmark; klara testet ≠ förståelse.
**Creative AI Applications** — Konst, design, musik och speltillgångar med generativa modeller som medskapande verktyg. Debatter om upphovsrätt.
**Generative AI Copyright** — Vem äger AI-genererat verk och fair use av träningsdata? EU AI Act och pågående rättsprocesser påverkar branschen.

Total terms: 2000
