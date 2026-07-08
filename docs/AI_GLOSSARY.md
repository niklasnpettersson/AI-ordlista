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

**Supervised Learning** — Lärande från märkta exempel där modellen optimerar en förlustfunktion mot kända målvariabler.
**Unsupervised Learning** — Lärande utan etiketter; modellen upptäcker struktur, kluster eller latenta representationer i data.
**Semi-supervised Learning** — Kombinerar liten mängd märkt data med stor omärkt datamängd för att förbättra generalisering.
**Self-supervised Learning** — Modellen skapar egna träningsmål från rådata, t.ex. maskerad token-prediktion eller kontrastiv inlärning.
**Transfer Learning** — Återanvänder representationer från en förtränad modell på en ny uppgift med mindre data eller beräkning.
**Domain Adaptation** — Anpassar en modell tränad i en domän (källdomän) till en annan (måldomän) med skiftad datadistribution.
**Feature Engineering** — Manuell eller regelbaserad konstruktion av indatarepresentationer som fångar relevant signal för modellen.
**Feature Selection** — Urval av delmängd av features som minskar dimensionalitet och overfitting utan att förlora prediktiv kraft.
**Bias-Variance Tradeoff** — Balans mellan modellens systematiska fel (bias) och känslighet för träningsvariation (varians).
**Overfitting** — Modellen memorerar träningsdata och presterar dåligt på osedd data; ofta tecken på för hög kapacitet.
**Underfitting** — Modellen är för enkel för att fånga datamönster; högt fel både på träning och validering.
**Regularization** — Tekniker som begränsar modellkomplexitet (L1/L2, dropout, early stopping) för bättre generalisering.
**Cross-validation** — Uppdelning av data i flera fold för robust uppskattning av generaliseringsprestanda.
**k-fold Cross-validation** — Data delas i k delar; varje fold används en gång som validering medan övriga tränar.
**Hold-out Validation** — En fast tränings-/valideringsdelning utan rotation; snabb men mer variabel uppskattning.
**Train/Validation/Test Split** — Tre väggar: träning för parametrar, validering för hyperparametrar, test för slutgiltig utvärdering.
**Hyperparameter** — Konfigurationsvärde satt före träning (learning rate, batch size) som inte lärs via gradient.
**Hyperparameter Tuning** — Systematisk sökning (grid, random, Bayesian) efter optimala hyperparametrar.
**Grid Search** — Exhaustiv utvärdering av fördefinierade hyperparameterkombinationer.
**Random Search** — Slumpmässigt urval av hyperparameterkombinationer; ofta effektivare än grid search i höga dimensioner.
**Bayesian Optimization** — Sekventiell modellbaserad sökning som balanserar utforskning och exploatering av hyperparameterrum.
**Learning Curve** — Graf över modellprestanda vs träningsdatamängd; avslöjar under/overfitting och datanhunger.
**Confusion Matrix** — Tabell över sanna/falska positiva/negativa för klassificering; grund för många mätvärden.
**Precision** — Andel predikterade positiva som faktiskt är positiva: TP/(TP+FP).
**Recall** — Andel faktiska positiva som hittas: TP/(TP+FN); även känt som sensitivity.
**F1 Score** — Harmoniskt medelvärde av precision och recall; balanserar båda när klasser är obalanserade.
**ROC Curve** — Plot av true positive rate mot false positive rate vid varierande klassificeringströsklar.
**AUC-ROC** — Area under ROC-kurvan; sammanfattande mått på diskrimineringsförmåga oberoende av tröskel.
**PR Curve** — Precision-recall-kurva; mer informativ än ROC vid stark klassobalans.
**AUC-PR** — Area under precision-recall-kurvan.
**Log Loss** — Korsentropiförlust för sannolikhetsutsagor; straffar överdrivet självsäkra felprediktioner.
**Calibration** — I hur hög grad predikterade sannolikheter matchar empiriska frekvenser.
**Platt Scaling** — Logistisk regression på modellens råa scores för att kalibrera sannolikheter.
**Isotonic Regression** — Icke-parametrisk monoton kalibrering som passar flexibla sannolikhetsmappningar.
**Class Imbalance** — Ojämn fördelning av klasser som kan snedvrida träning och mätvärden.
**Oversampling** — Ökar representationen av minoritetsklasser, t.ex. via duplication eller SMOTE.
**Undersampling** — Minskar majoritetsklasser för balans; risk att förlora information.
**SMOTE** — Synthetic Minority Over-sampling Technique; interpolerar syntetiska minoritetsexempel.
**Cost-sensitive Learning** — Viktar förlust efter klasskostnad för att hantera obalans eller asymmetriska fel.
**Ensemble Learning** — Kombinerar flera modeller för robustare prediktioner än enskilda modeller.
**Bagging** — Bootstrap aggregating; tränar parallella modeller på bootstrap-sampel och aggregerar.
**Boosting** — Sekventiell ensemble som fokuserar på tidigare modellers fel, t.ex. gradient boosting.
**Stacking** — Meta-modell lär sig kombinera basmodellers prediktioner.
**Random Forest** — Ensemble av beslutsträd med bagging och slumpmässigt feature-urval per split.
**Gradient Boosting** — Sekventiellt adderar träd som approximerar negativ gradient av förlust.
**XGBoost** — Optimerad gradient boosting-implementation med regularisering och effektiv träning.
**LightGBM** — Gradient boosting med leaf-wise trädtillväxt och histogram-baserad split-sökning.
**CatBoost** — Gradient boosting med inbyggd hantering av kategoriska features och ordningsbevarande encoding.
**Decision Tree** — Rekursiv partitionering av feature-rymden via binära split-regler.
**Random Split** — Slumpmässig uppdelning av features vid varje nod i random forest.
**Support Vector Machine** — Hittar maximal marginal-hyperplan; kan utökas med kernel-trick för icke-linjära gränser.
**Kernel Trick** — Implicit mappning till högdimensionellt rum utan explicit feature-transformation.
**k-Nearest Neighbors** — Icke-parametrisk metod som klassificerar/regresserar baserat på närmaste träningspunkter.
**Naive Bayes** — Probabilistisk klassificerare som antar feature-oberoende givet klass.
**Logistic Regression** — Linjär modell med logistisk länk för binär/multinomial klassificering.
**Linear Regression** — Modellerar kontinuerligt mål som linjär kombination av features med minsta kvadrat.
**Ridge Regression** — Linjär regression med L2-regularisering som krymper koefficienter.
**Lasso Regression** — Linjär regression med L1-regularisering som ger sparsity och feature selection.
**Elastic Net** — Kombinerar L1- och L2-straff för balanserad regularisering.
**Principal Component Analysis** — Linjär dimensionsreduktion som projicerar till ortogonala riktningar med maximal varians.
**t-SNE** — Icke-linjär embeddingsmetod för visualisering av högdimensionell data i 2D/3D.
**UMAP** — Manifold learning för dimensionsreduktion; bevarar både lokal och global struktur bättre än t-SNE i många fall.
**K-means Clustering** — Partitionerar data i k kluster genom iterativ uppdatering av centroid och tilldelning.
**Hierarchical Clustering** — Bygger klusterträd via agglomerativ eller divisiv länkning.
**DBSCAN** — Densitetsbaserad klustring som hittar godtyckligt formade kluster och markerar brus som outliers.
**Gaussian Mixture Model** — Probabilistisk klustring som modellerar data som blandning av Gaussiska komponenter.
**Expectation-Maximization** — Iterativ algoritm för att uppskatta latenta variabler i modeller som GMM.
**Curse of Dimensionality** — Fenomen där data blir gles i höga dimensioner och avståndsmått förlorar discriminativ kraft.
**Curriculum Learning** — Träningsstrategi som presenterar exempel i ökande svårighetsordning.
**Active Learning** — Modellen väljer vilka exempel som ska annoteras för maximal informationsvinst.
**Weak Supervision** — Använder brusiga, heuristiska eller partiella etiketter istället för full manuell annotation.
**Label Noise** — Felaktiga eller inkonsistenta etiketter i träningsdata som degraderar inlärning.
**Data Augmentation** — Syntetisk utökning av träningsdata via transformationer som bevarar semantik.
**Synthetic Data** — Artificiellt genererad data för träning, ofta via simulering eller generativa modeller.
**Concept Drift** — Förändring av datadistribution eller målrelation över tid i produktion.
**Covariate Shift** — Förändring av indatafördelning P(X) medan P(Y|X) är stabil.
**Label Shift** — Förändring av målfördelning P(Y) medan P(X|Y) är stabil.
**Prior Shift** — Förändring av klasspriorer mellan träning och deployment.
**Distribution Shift** — Generell term för mismatch mellan tränings- och test-/produktionsfördelning.
**OOD Detection** — Out-of-distribution detection; identifierar indata som avviker från träningsfördelningen.
**Anomaly Detection** — Hittar avvikande observationer som inte följer normalt mönster.
**One-class SVM** — Lär en gräns runt normal data för att flagga avvikelser.
**Isolation Forest** — Anomaly detection genom att isolera observationer med få slumpmässiga splits.
**Autoencoder** — Neuralt nätverk som kodar och avkodar data; rekonstruktionsfel indikerar anomalier.
**Inductive Bias** — Inbyggda antaganden i modellarkitektur eller algoritm som styr vilka funktioner som lärs lätt.
**No Free Lunch Theorem** — Inget universellt bästa inlärningsalgoritm över alla möjliga datadistributioner.
**Sample Complexity** — Antal exempel som krävs för att lära en hypotesklass med given noggrannhet.
**PAC Learning** — Probably Approximately Correct framework för inlärbarhet med sannolikhetsgarantier.
**VC Dimension** — Mått på modellklassens kapacitet relaterat till generaliseringsgränser.
**Rademacher Complexity** — Empiriskt mått på hypotesklassens rikedom och risk för overfitting.
**Generalization Gap** — Skillnad mellan tränings- och testprestanda; indikator på overfitting.
**Double Descent** — Fenomen där testfel kan minska igen efter interpolationspunkten när modellen växer.
**Interpolation Regime** — Träningsregim där modellen kan interpolera all träningsdata (noll träningsfel).
**Scaling Laws** — Empiriska potenslagar som beskriver hur prestanda skalar med modellstorlek, data och compute.
**Data-centric AI** — Fokus på att förbättra datakvalitet, annotation och pipeline snarare än enbart modellarkitektur.
**Leakage** — Oavsiktlig exponering av test-/målinformation under träning som ger optimistiska mätvärden.
**Target Leakage** — Features som indirekt innehåller målvariabeln och inte finns vid prediktionstid.
**Train-test Contamination** — Överlapp mellan tränings- och testdata som snedvrider utvärdering.
**Reproducibility** — Förmåga att återskapa resultat med samma kod, data och seed.
**Ablation Study** — Systematisk borttagning av komponenter för att mäta deras bidrag till prestanda.

## Deep Learning

**Neural Network** — Beräkningsgraf av sammankopplade neuroner som approximerar icke-linjära funktioner via viktade transformationer.
**Deep Neural Network** — Nätverk med många dolda lager som lär hierarkiska representationer.
**Perceptron** — Enkel binär klassificerare: viktad summa följt av tröskelaktivering.
**Multilayer Perceptron** — Fully connected feedforward-nätverk med ett eller flera dolda lager.
**Activation Function** — Icke-linjär transformation efter linjärt lager; möjliggör universell approximation.
**ReLU** — Rectified Linear Unit: max(0,x); standardaktivering tack vare snabbhet och gradientflöde.
**Leaky ReLU** — ReLU-variant som tillåter liten negativ slope för att undvika döda neuroner.
**GELU** — Gaussian Error Linear Unit; mjuk gating använd i transformers, approximerar x·Φ(x).
**SiLU / Swish** — x·σ(x); mjuk icke-linearitet som ofta presterar bättre än ReLU i djupa nätverk.
**Sigmoid** — S-formad aktivering som mappar till (0,1); används historiskt och i gating.
**Tanh** — Hyperbolisk tangens; centrerad sigmoid med output i (-1,1).
**Softmax** — Normaliserar logits till sannolikhetsfördelning över klasser.
**Logits** — Råa osignerade scores före softmax eller sigmoid.
**Weight Initialization** — Startvärden för vikter som påverkar konvergens och gradientstabilitet.
**Xavier Initialization** — Varians skalad för att bevara aktiveringsvarians genom lager (tanh/sigmoid).
**He Initialization** — Varians skalad med fan-in för ReLU-nätverk.
**Backpropagation** — Effektiv beräkning av gradienter via kedjeregeln genom beräkningsgrafen.
**Computational Graph** — DAG som representerar operationer och möjliggör automatisk differentiering.
**Automatic Differentiation** — Exakt gradientberäkning via symbolisk/registrerad graf snarare än numerisk approximation.
**Forward Pass** — Beräkning av outputs genom nätverket givet indata.
**Backward Pass** — Gradientberäkning från förlust tillbaka genom lagren.
**Gradient** — Partiell derivata av förlust w.r.t. parametrar; styr uppdateringsriktning.
**Gradient Descent** — Iterativ parameteruppdatering i negativ gradientriktning.
**Stochastic Gradient Descent** — SGD med gradient estimerad från minibatch istället för hela datasetet.
**Minibatch** — Delmängd av träningsexempel per uppdateringssteg; balanserar brus och effektivitet.
**Batch Size** — Antal exempel per gradientsteg; påverkar minne, brus och generalisering.
**Learning Rate** — Stegstorlek i parameteruppdatering; kritisk hyperparameter för konvergens.
**Learning Rate Schedule** — Tidsvarierande learning rate (cosine, step decay, warmup) för stabil träning.
**Warmup** — Gradvis ökning av learning rate i början av träning för att undvika instabilitet.
**Cosine Annealing** — Learning rate som följer cosinuskurva mot minimum.
**Momentum** — Ackumulerar velocity i gradientriktning för snabbare och stabilare konvergens.
**Nesterov Momentum** — Momentum som tittar framåt; ofta bättre konvergens än klassisk momentum.
**Adam** — Adaptiv optimizer som kombinerar momentum och per-parameter learning rates.
**AdamW** — Adam med korrekt decoupled weight decay; standard i transformer-träning.
**RMSprop** — Adaptiv optimizer som skalar learning rate med rullande gradientkvadratmedel.
**Adagrad** — Adaptiv optimizer med ackumulerad gradientkvadrat; learning rate minskar över tid.
**Lion** — Memory-efficient optimizer som använder tecken av momentum för uppdatering.
**Weight Decay** — L2-regularisering som krymper vikter vid varje steg.
**Gradient Clipping** — Begränsar gradientnorm för att förhindra explosiva uppdateringar.
**Vanishing Gradient** — Gradienter som exponentiellt krymper i djupa nätverk och hindrar inlärning i tidiga lager.
**Exploding Gradient** — Gradienter som växer okontrollerat; kan orsaka numerisk instabilitet.
**Batch Normalization** — Normaliserar aktiveringar per batch; stabiliserar träning och tillåter högre learning rates.
**Layer Normalization** — Normaliserar över feature-dimension per exempel; standard i transformers.
**Group Normalization** — Normaliserar kanalgrupper; robust när batch size är liten.
**Instance Normalization** — Normaliserar per kanal och spatial position; vanligt i style transfer.
**RMSNorm** — Root Mean Square normalization; förenklad variant utan mean-centering, använd i LLaMA m.fl.
**Dropout** — Slumpmässigt nollställer neuroner under träning som implicit ensemble.
**DropConnect** — Slumpmässigt nollställer vikter istället för neuroner.
**Stochastic Depth** — Slumpmässigt hoppar över residual block under träning.
**Residual Connection** — Skip connection som adderar input till output: y = F(x) + x; möjliggör djupa nätverk.
**Highway Network** — Gated skip connections som lär sig hur mycket transformation vs identitet som ska passera.
**DenseNet** — Varje lager tar input från alla föregående lager; stark feature-återanvändning.
**Convolutional Layer** — Tillämpar lärd filter över lokala receptive fields för att fånga spatiala mönster.
**Kernel / Filter** — Liten viktmatris som convolveras över indata.
**Stride** — Stegstorlek då filtret förflyttas; större stride minskar spatial upplösning.
**Padding** — Tillägg av border-värden för att kontrollera output-storlek.
**Receptive Field** — Region i indata som påverkar en given neurons aktivering.
**Pooling Layer** — Aggregerar spatial information (max/avg) för dimensionsreduktion och translation invariance.
**Max Pooling** — Tar maximum inom varje poolingsfönster.
**Average Pooling** — Tar medelvärde inom poolingsfönster.
**Global Average Pooling** — Medelvärde över hela spatial dimension; ersätter ofta FC-lager i CNNs.
**Transposed Convolution** — Upsampling-lager som lär sig spatial uppskalning; används i decoders och GANs.
**Dilated Convolution** — Convolution med hoppade kernel-positioner för större receptive field utan mer parametrar.
**Depthwise Separable Convolution** — Faktoriserar convolution i depthwise + pointwise; effektivare i mobilarkitekturer.
**Recurrent Neural Network** — Nätverk med feedback-loopar för sekvensmodellering.
**LSTM** — Long Short-Term Memory; gated RNN som hanterar långsiktiga beroenden.
**GRU** — Gated Recurrent Unit; förenklad LSTM-variant med färre parametrar.
**Bidirectional RNN** — Processar sekvens i båda riktningar; kräver hel sekvens tillgänglig.
**Seq2Seq** — Encoder-decoder-arkitektur som mappar en sekvens till en annan.
**Teacher Forcing** — Under träning matas decoder med ground truth tokens istället för egna prediktioner.
**Exposure Bias** — Mismatch mellan träning (teacher forcing) och inferens (autoregressiv sampling).
**Attention Mechanism** — Dynamisk viktning av relevanta delar av input; kärnan i transformers.
**Self-Attention** — Attention där queries, keys och values kommer från samma sekvens.
**Multi-Head Attention** — Parallella attention-huvuden som fångar olika relationstyper.
**Positional Encoding** — Injicerar positionsinformation i sekvensrepresentationer.
**Sinusoidal Positional Encoding** — Fast sin/cos-kodning av position från original transformer-papperet.
**Learned Positional Embedding** — Tränade positionvektorer istället för fast encoding.
**Rotary Position Embedding** — RoPE; roterar query/key-vektorer för relativ positionskodning.
**ALiBi** — Attention with Linear Biases; extrapolerar till längre sekvenser utan explicit positionsembedding.
**Flash Attention** — IO-aware exakt attention-algoritm som minskar HBM-åtkomst och minneskrav.
**Mixed Precision Training** — Träning med FP16/BF16 för hastighet kombinerat med FP32 master weights.
**BF16** — Brain Float 16; bredare exponent än FP16, ofta mer stabilt för deep learning.
**FP16** — 16-bitars flyttal; halverar minne men risk för underflow/overflow.
**TF32** — NVIDIA-format som accelererar matmul med reducerad precision internt.
**Gradient Accumulation** — Ackumulerar gradienter över flera microbatches för effektiv större batch size.
**Gradient Checkpointing** — Rekomputera activations vid backward pass för att spara minne.
**Knowledge Distillation** — Mindre student-modell lär sig matcha större teacher-modells outputs eller logits.
**Model Compression** — Tekniker (pruning, quantization, distillation) för mindre/snabbare modeller.
**Pruning** — Borttagning av vikter/neuroner med liten påverkan på output.
**Quantization** — Reducerar numerisk precision (INT8/INT4) för inferens eller träning.
**Post-training Quantization** — Kvantisering efter träning utan finjustering.
**Quantization-aware Training** — Simulerar kvantisering under träning för bättre INT-precision.
**LoRA** — Low-Rank Adaptation; tränar lågrangsuppdateringar av vikter istället för full fine-tuning.
**Parameter-efficient Fine-tuning** — PEFT; familj av metoder som uppdaterar liten delmängd av parametrar.
**Neural Architecture Search** — Automatiserad sökning efter optimal nätverksarkitektur.
**Universal Approximation Theorem** — Teoretiskt resultat: tillräckligt brett MLP kan approximera kontinuerliga funktioner.
**Mode Collapse** — Generativ modell producerar begränsad variation av outputs.
**Dead ReLU** — Neuron vars vikt aldrig uppdateras eftersom input alltid är negativ.
**Spectral Normalization** — Begränsar singularvärden av vikter för Lipschitz-stabilitet i GANs.
**Early Stopping** — Avbryter träning när valideringsförlust slutar förbättras; implicit regularisering.

## Transformers & Attention

**Self-Attention** — Attention där query, key och value hämtas från samma sekvens; möjliggör parallell kontextuell modellering utan rekurrens.
**Cross-Attention** — Attention där queries kommer från en sekvens och keys/values från en annan; central i encoder-decoder och multimodala modeller.
**Scaled Dot-Product Attention** — Beräknar softmax(QK^T/√d_k)V; skalningen stabiliserar gradienter när dimensionsstorleken växer.
**Multi-Head Attention** — Parallella attention-huvuden projicerar Q/K/V till delrum och aggregerar flera relationstyper.
**Query-Key-Value Projection** — Linjära projektioner som mappar dolda tillstånd till Q, K och V för attention-beräkning.
**Causal Mask** — Triangulär mask som förhindrar att positioner ser framtida tokens; krävs för autoregressiv generering.
**Bidirectional Attention** — Fullständig token-till-token-interaktion utan kausal begränsning; används i encoders som BERT.
**Encoder-Decoder Attention** — Decoder-queries attendar över encoder-keys/values; kopplar käll- och målsekvenser.
**Pre-Norm Transformer** — Layer normalization placeras före attention/FFN; ofta stabilare träning i djupa modeller.
**Post-Norm Transformer** — Original transformer-layout med normalisering efter sub-lager; kräver ofta warmup.
**Feed-Forward Network Block** — Position-wise två-lagers MLP med icke-linearitet mellan attention-lager; ~2/3 av transformer-parametrar.
**Position-Wise FFN** — Samma MLP tillämpas oberoende på varje token-position utan parameterdelning över sekvensen.
**Sinusoidal Positional Encoding** — Fast sin/cos-kodning av absolut position; generaliserar delvis till längre sekvenser än träningslängd.
**Learned Positional Embedding** — Tränade positionvektorer adderas till token-embedding; enkelt men begränsat av max-sekvenslängd.
**Rotary Position Embedding** — RoPE roterar Q/K i komplexplan baserat på position; kodar relativ position naturligt.
**Relative Position Bias** — Additiv bias till attention-scores baserad på tokenavstånd; fångar lokal struktur utan absolut embedding.
**ALiBi** — Attention with Linear Biases: avståndsbaserade straff i attention utan explicit positionsembedding; bra extrapolation.
**Flash Attention** — IO-aware blockvis attention som minimerar HBM-läs/skriv; exakt men betydligt snabbare och minnessnålare.
**FlashAttention-2** — Förbättrad parallellisering och arbetsfördelning i Flash Attention; högre GPU-utnyttjande.
**PagedAttention** — Virtuell minneshantering för KV-cache i block; möjliggör effektiv batched inferens med varierande sekvenslängder.
**KV Cache** — Cachade key/value-tensorer från tidigare tokens vid autoregressiv inferens; undviker omräkning.
**Multi-Query Attention** — MQA: delade K/V-projektioner över huvuden; minskar KV-cache-minne vid inferens.
**Grouped-Query Attention** — GQA: mellanting mellan MHA och MQA; grupper av huvuden delar K/V för minnes-/kvalitetsbalans.
**Sliding Window Attention** — Varje token attendar endast inom lokalt fönster; linjärt minne i sekvenslängd.
**Longformer Attention** — Kombinerar lokalt fönster med få globala tokens för långkontext utan full kvadratisk kostnad.
**BigBird Attention** — Sparse mönster med random, window och global tokens; teoretisk universal-approximator-garanti.
**Linformer Attention** — Lågranksaproximation av attention-matris via projicerade keys; O(n) i sekvenslängd.
**Performer Attention** — Approximerar softmax-kernel med FAVOR+ för linjärt minne och tid.
**Linear Attention** — Ersätter softmax med kernel-trick som tillåter associativ beräkning; sub-kvadratisk.
**Sparse Attention Pattern** — Fördefinierat gles attention-mönster som minskar beräkningskostnad för långa sekvenser.
**Local Attention** — Begränsad receptive field per token; används i bild-transformers och långtext.
**Global Attention Tokens** — Utvalda tokens med full sekvensåtkomst som agerar informationshubbar.
**Attention Dropout** — Slumpmässig nollställning av attention-vikter efter softmax; regularisering under träning.
**Attention Sink** — Tendens att första token får oproportionerligt hög attention-vikt; relevant vid prompt-cache.
**Induction Head** — Attention-mekanism som kopierar mönster från tidigare kontext; kopplat till in-context learning.
**Transformer Block** — Standard enhet: attention + residual + FFN + residual, eventuellt med normalisering.
**Encoder-Only Transformer** — Stack av self-attention-block utan kausal mask; för förståelseuppgifter.
**Decoder-Only Transformer** — Kausalt maskerad stack; grund för moderna LLM:er.
**Encoder-Decoder Transformer** — Separata encoder och decoder med cross-attention; seq2seq och översättning.
**Token Mixing** — Kombination av attention (tokenmixning) och FFN (kanalmixning) i vision transformers.
**Patch Embedding** — Delar upp bild i patchar som linjärt projiceras till token-vektorer; ViT-ingång.
**CLS Token** — Special token vars representation aggregerar sekvensinformation för klassificering.
**Sequence Packing** — Packar flera korta sekvenser i en batch-rad med separata attention-masker; högre GPU-utnyttjande.
**Context Parallelism** — Delar lång sekvens över enheter med kommunikation vid attention-gränser.
**Tensor Parallelism in Attention** — Delar QKV-projektioner och attention-beräkning horisontellt över GPU:er.
**Ring Attention** — Distribuerar KV-block i ring-topologi för att träna extremt långa sekvenser.
**NTK-Aware Scaling** — Skalar RoPE-bas för att interpolera till längre kontext utan finjustering.
**YaRN** — Yet another RoPE extensioN: finjusterar frekvensinterpolering för längre kontext med minimal degradering.
**Position Interpolation** — Komprimerar position-index vid inferens för att passa tränad max-längd; enkel kontextförlängning.
**Extrapolation vs Interpolation** — Extrapolation testar längre sekvenser direkt; interpolation mappar positioner till träningsintervall.
**Attention Temperature** — Skalning av QK^T före softmax; högre temperatur ger mjukare fördelning.
**Softmax Numerical Stability** — Subtraherar max(QK^T) före exp för att undvika overflow i attention.
**Disentangled Attention** — DeBERTa: separerar innehålls- och positionsrepresentationer i Q/K.
**Talking-Heads Attention** — Extra linjära transformationer före/efter softmax över huvuden.
**Multi-Scale Attention** — Kombinerar attention över flera upplösningar eller fönsterstorlekar.
**Cross-Layer Attention** — Attention mellan representationer från olika lager; sällsynt men används i vissa arkitekturer.
**Memory-Augmented Transformer** — Externa minnesmoduler som tokens kan attenda över utöver aktuell sekvens.
**Transformer-XL Segment Recurrence** — Cachar dolda tillstånd från föregående segment för längre beroenden.
**Compressive Transformer** — Komprimerar äldre minne till sammanfattningsvektorer för längre effektiv kontext.
**Perceiver IO** — Latent bottleneck som attendar över indata av godtycklig storlek/form; generisk multimodal encoder.
**Perceiver Resampler** — Fast antal latenta vektorer som komprimerar variabel indata till fix representation.
**Attention Map Visualization** — Heatmap över attention-vikter; diagnostiskt men tolkning kräver försiktighet.
**Head Pruning** — Borttagning av attention-huvuden med litet bidrag till downstream-prestanda.
**Attention Entropy** — Entropi av attention-fördelning; låg entropi indikerar skarp fokusering.
**Causal Language Modeling Head** — Linjärt lager + softmax som predicerar nästa token från dolda tillstånd.
**Masked Language Modeling Head** — Predicerar maskerade tokens från bidirectional kontext; BERT-style.
**Attention as Soft Dictionary Lookup** — Tolkning: softmax-viktad summa av values som lookup via key-matchning.
**Low-Rank Attention Approximation** — Faktoriserar attention-matris för att minska beräkning och minne.
**Blockwise Parallel Attention** — Beräknar attention i block parallellt; grund för Flash Attention.
**Online Softmax** — Inkrementell softmax-beräkning över block utan full materialisering av score-matris.
**Stochastic Attention** — Slumpmässigt subsampling av keys/values under träning som regularisering.
**Cross-Document Attention** — Attention över flera dokument i samma sekvens; kräver noggrann maskering.
**Document-Level Attention Mask** — Separerar attention inom dokument för att undvika otillåten cross-doc-leakage.
**Prefix Attention Mask** — Tillåter full bidirectional attention inom prompt-prefix och kausal generering efter.
**U-Net Transformer Hybrid** — Kombinerar U-Net-skip med transformer-block i diffusion och vision.
**Swin Shifted Window** — Shifted window attention i hierarkisk vision transformer för cross-window-koppling.
**Axial Attention** — Factoriserar 2D-attention till rad- och kolumnpass; effektiv för bilder.
**Factorized Attention** — Delar upp full attention i faktoriserade steg över dimensioner eller positioner.
**Token Dropping in Attention** — Dynamiskt hoppar över tokens med låg attention-saliency för acceleration.
**Query-Key Normalization** — QK-norm stabiliserar attention i stora modeller genom att normalisera projektioner.
**Attention Logit Capping** — Begränsar max attention-score för att förhindra dominans av enskilda keys.
**Relative Attention Core** — Generellt ramverk för att injicera strukturell prior via relativ position/innehåll.
**Transformer Depth vs Width** — Tradeoff mellan antal lager och dold dimension; påverkar kapacitet och parallellism.
**Attention FLOPs Scaling** — Attention-kostnad skalar O(n²·d) i sekvenslängd n och dimension d.
**Memory Bandwidth Bottleneck** — Attention ofta minnesbandbreddsbegränsad snarare än compute-bound på GPU.
**Speculative Decoding with KV Reuse** — Draft-modell delar eller approximerar KV-cache för snabbare validering.
**Multi-Token Prediction Head** — Auxiliary heads som predicerar flera framtida tokens parallellt; accelererar träning.
**Mixture-of-Depths** — Dynamiskt hoppar över lager per token baserat på router; sparar compute.
**LayerDrop** — Slumpmässigt droppar hela transformer-lager under träning som regularisering.
**Sandwich Normalization** — Extra normalisering runt sub-lager; vissa arkitekturer för stabil deep training.

## Stora språkmodeller

**Large Language Model** — Skalad autoregressiv transformer tränad på massiv textkorpus; generaliserar brett via nästa-token-prediktion.
**Autoregressive Generation** — Sekventiell token-generering där varje steg conditionar på alla tidigare tokens.
**Next-Token Prediction** — Träningsmål: maximera sannolikhet för korrekt nästa token givet prefix.
**Tokenization** — Mappning text→diskreta token-ID via BPE, WordPiece eller SentencePiece.
**Byte-Pair Encoding** — BPE: iterativt slår samman frekventa bytepar till subword-vokabulär.
**SentencePiece** — Language-agnostic tokenisering direkt på rå bytes/Unicode utan förbehandling.
**Vocabulary Size** — Antal unika tokens; större vokabulär minskar sekvenslängd men ökar embedding-parametrar.
**Context Window** — Max antal tokens modellen kan processa i en forward pass; begränsar prompt+generering.
**Long-Context LLM** — Modell optimerad för kontext >> träning via arkitektur, data eller finjustering.
**Emergent Abilities** — Förmågor som dyker upp vid skalning och inte var förutsägbara från mindre modeller.
**In-Context Learning** — Modellen löser nya uppgifter via exempel i prompten utan gradientuppdatering.
**Few-Shot Prompting** — Ger några demonstrations-exempel i prompten för att styra beteende.
**Zero-Shot Prompting** — Instruktion utan exempel; förlitar sig på förtränad kunskap och instruktionsföljning.
**Chain-of-Thought** — CoT: modellen genererar mellansteg som förbättrar resonemang på komplexa frågor.
**System Prompt** — Fast instruktion som sätter roll, ton och begränsningar för hela konversationen.
**Prompt Engineering** — Design av indataformulering för att maximera kvalitet utan modelländring.
**Temperature Sampling** — Skalar logits före sampling; högre T ger mer variation, lägre T mer deterministiskt.
**Top-k Sampling** — Begränsar sampling till k mest sannolika tokens vid varje steg.
**Top-p Nucleus Sampling** — Sample från minsta tokenmängd vars kumulativa sannolikhet ≥ p.
**Min-p Sampling** — Filtrerar tokens under dynamisk sannolikhetströskel relativt top-token.
**Repetition Penalty** — Straffar logits för nyligen genererade tokens för att minska loopar.
**Frequency Penalty** — Linjärt straff per token baserat på historisk frekvens i genererad text.
**Presence Penalty** — Engångsstraff om token redan förekommit; uppmuntrar nytt innehåll.
**Beam Search** — Bevarar flera hypoteser parallellt; vanligt i maskinöversättning, sällan i chat-LLM.
**Greedy Decoding** — Väljer argmax-token varje steg; snabbt men ofta repetitivt och suboptimalt.
**Stop Sequence** — Special token eller sträng som avslutar generering när den produceras.
**Max Tokens** — Övre gräns på genererade tokens; kontrollerar kostnad och latens.
**Prefill Phase** — Initial forward pass som processar hela prompten och bygger KV-cache.
**Decode Phase** — Autoregressiva steg som appendar en token i taget efter prefill.
**Time to First Token** — TTFT: latens från request till första genererade token; domineras av prefill.
**Tokens Per Second** — Genomströmning i decode-fas; nyckelmetrik för inferensprestanda.
**Mixture of Experts LLM** — MoE-LLM: sparsam aktivering av expert-FFN per token för skalad kapacitet.
**Expert Routing** — Router väljer top-k experter per token; load balancing är träningsutmaning.
**Dense vs MoE LLM** — Dense aktiverar alla parametrar; MoE aktiverar delmängd vid inferens.
**Chinchilla Optimal** — Empiriskt optimalt förhållande compute/data/modellstorlek enligt scaling laws.
**Compute-Optimal Training** — Träningsbudget allokeras för att balansera modellstorlek och tokens enligt scaling law.
**Data Contamination** — Benchmark-data i träningskorpus som inflates utvärderingsresultat.
**Memorization vs Generalization** — LLM kan memorerar sekvenser vs lära generaliserbara mönster; svårt att skilja.
**Hallucination** — Generering av plausible men faktiskt felaktigt innehåll med hög konfidens.
**Confabulation** — Synonym till hallucination; modellen fyller i luckor med fabricerade detaljer.
**Instruction Tuning** — Finjustering på (instruktion, svar)-par för bättre användarlydnad.
**Chat Template** — Formateringskonvention som wrappar meddelanden i special tokens för konversationsmodeller.
**Base Model vs Instruct Model** — Base är rå förträning; instruct är finjusterad för dialog och uppgiftsföljning.
**Continued Pretraining** — Extra förträning på domänspecifik korpus före downstream fine-tuning.
**Mid-Training** — Mellansteg med curated data mellan massiv pretrain och alignment.
**Synthetic Instruction Data** — AI-genererade instruktionssvar för att skala finjusteringsdata.
**Self-Instruct** — Modell genererar egna instruktioner och svar som träningsdata.
**Constitutional AI Data** — Självkritik och revision enligt principer för att producera alignment-data.
**Token Healing** — Reparera tokengränser vid streaming så att ofullständiga UTF-8-sekvenser inte bryts.
**Logit Bias** — Additiv offset på specifika token-logits för att styra output under inferens.
**Structured Output Mode** — Tvingar JSON/schema via constrained decoding eller grammar guidance.
**JSON Mode** — API-läge som instruerar modellen att producera giltig JSON.
**Function Calling** — Modellen väljer verktyg och argument i strukturerat format för exekvering.
**Tool Use** — LLM delegerar beräkning, sökning eller API-anrop till externa verktyg.
**Agentic Loop** — Iterativ cykel: planera → verktyg → observera → uppdatera tills mål uppnås.
**Reasoning Model** — Modell tränad/finjusterad för längre intern resonemang före svar (o1-liknande).
**Test-Time Compute Scaling** — Allokerar mer inferens/compute vid svar för bättre kvalitet (best-of-N, tree search).
**Best-of-N Sampling** — Genererar N kandidater och väljer bästa via verifierare eller reward model.
**Process Reward Model** — PRM: betygsätter mellansteg i resonemang, inte bara slutresultat.
**Outcome Reward Model** — ORM: betygsätter slutligt svar baserat på korrekthet eller preferens.
**Model Merging** — Kombinerar vikter från flera finjusterade modeller (SLERP, TIES, DARE).
**Quantized LLM** — LLM i INT4/INT8/FP8 för minskat minne och snabbare inferens.
**GGUF Format** — Kvantiserat modellformat för lokal inferens med llama.cpp.
**Speculative Decoding** — Liten draft-modell föreslår tokens som stor modell verifierar parallellt.
**Medusa Heads** — Extra decoding-huvuden för parallell multi-token-prediktion vid inferens.
**Prompt Caching** — Återanvänder KV-cache för identiska prompt-prefix; sänker latens och kostnad.
**Prefix Caching** — Server-side cache av beräknade prefix-representationer mellan requests.
**Batch Inference** — Processar flera prompts parallellt med padding/packing för GPU-effektivitet.
**Continuous Batching** — Dynamiskt lägger till/avslutar sekvenser i pågående batch (iteration-level scheduling).
**Disaggregated Prefill-Decode** — Separata workers för compute-tung prefill vs latens-känslig decode.
**Model Parallelism for LLM** — Tensor/pipeline/expert parallellism för att träna/serva modeller > en GPU.
**Pipeline Parallelism** — Delar lager över enheter; microbatching för att hålla pipeline full.
**Expert Parallelism** — Distribuerar MoE-experter över noder med all-to-all kommunikation.
**Sequence Parallelism** — Delar sekvensdimension över enheter vid lång kontext-träning.
**FSDP** — Fully Sharded Data Parallel: shardar parametrar, gradienter och optimizer-states.
**ZeRO Optimization** — DeepSpeed Zero Redundancy Optimizer; minskar minnesduplicering i distribuerad träning.
**Activation Checkpointing** — Rekomputera activations i backward för att träna större modeller på samma minne.
**Loss Spike** — Plötslig ökning av träningsförlust; kan indikera instabilitet eller dålig batch.
**Loss Divergence** — Förlust växer okontrollerat; ofta learning rate eller numerisk instabilitet.
**Learning Rate Warmup** — Linjärt ökar LR från nära noll; kritisk för stabila LLM-pretrain.
**Weight Tying** — Delar vikter mellan input embedding och output LM-head; minskar parametrar.
**Tied Embeddings** — Samma matris för token lookup och logits-projektion.
**Vocabulary Expansion** — Lägger till tokens för kod, språk eller specialtecken efter initial träning.
**Embedding Resizing** — Utökar embedding-matris vid ny vokabulär; kräver ofta finjustering.
**Tokenizer Mismatch** — Fel när tränings- och inferens-tokenizer skiljer sig; ger korruptionsartefakter.
**BPE Merge Table** — Ordning av subword-sammanslagningar som definierar tokenisering.
**Special Tokens** — Reserverade tokens för BOS, EOS, PAD, mask, verktyg och roller.
**BOS Token** — Beginning-of-sequence markerar start; ibland implicit i chat templates.
**EOS Token** — End-of-sequence signalerar att modellen ska sluta generera.
**PAD Token** — Padding till fix batch-längd; måste maskeras i attention.

## Fine-tuning & Adaptation

**Fine-Tuning** — Fortsatt träning av förtränade vikter på uppgiftsspecifik data med lägre learning rate.
**Full Fine-Tuning** — Uppdaterar alla modellparametrar; maximalt utrymme men dyrt i minne och risk för catastrophic forgetting.
**Parameter-Efficient Fine-Tuning** — PEFT: uppdaterar liten delmängd parametrar (LoRA, adapters) med jämförbar prestanda.
**LoRA** — Low-Rank Adaptation: tränar lågrangsmatriser ΔW=BA som adderas till frysta vikter.
**LoRA Rank** — Rang r i lågranksuppdatering; högre r ger mer kapacitet men fler träningsparametrar.
**LoRA Alpha** — Skalningsfaktor för LoRA-uppdatering; styr effektiv learning rate för adaptern.
**QLoRA** — Kvantiserad basmodell (4-bit) med LoRA-träning; möjliggör finjustering på konsument-GPU.
**DoRA** — Weight-Decomposed Low-Rank Adaptation; separerar magnitud och riktning för bättre stabilitet.
**AdaLoRA** — Adaptiv rangallokering under träning via SVD-baserad parameterbudget.
**IA3** — Infused Adapter: lär sig elementvisa skalningsvektorer istället för full lågranksmatris.
**Prefix Tuning** — Tränar kontinuerliga prefix-vektorer prepended till varje lager utan att ändra basvikter.
**Prompt Tuning** — Endast input-side soft prompts tränas; extremt parameter-effektivt.
**P-Tuning v2** — Djupa prompt-vektorer i varje lager; starkare än ytlig prompt tuning.
**Adapter Module** — Små bottleneck-lager insatta mellan transformer-block; task-specifik utan full finetune.
**Adapter Fusion** — Kombinerar flera adapters med viktning eller routing för multi-task.
**BitFit** — Finjusterar endast bias-termer; överraskande effektivt för vissa uppgifter.
**LayerNorm Tuning** — Uppdaterar endast normaliseringsparametrar för lätt domain shift.
**Head Tuning** — Finjusterar endast task-specifikt klassificerings/LM-head ovanpå fryst backbone.
**Selective Fine-Tuning** — Uppdaterar endast utvalda lager baserat på gradient/importance-analys.
**Catastrophic Forgetting** — Basförmågor försämras när modellen specialiseras på smal uppgift.
**Elastic Weight Consolidation** — EWC: straffar ändringar av viktiga parametrar via Fisher-information.
**Experience Replay** — Mixar gamla träningsdata vid kontinuerlig finjustering för att bevara kunskap.
**Multi-Task Fine-Tuning** — Tränar på flera uppgifter samtidigt för delad representation.
**Task Arithmetic** — Kombinerar task-vektorer (skill vectors) via addition/subtraktion i viktutrymmet.
**Model Soups** — Medelvärdar vikter från flera finetuning-körningar för robustare generalisering.
**WiSE-FT** — Interpolation mellan förtränade och finjusterade vikter för robusthet mot distribution shift.
**Domain-Adaptive Fine-Tuning** — Finjustering på måldomän efter generell pretrain; hanterar covariate shift.
**Instruction Fine-Tuning** — Finjustering på instruktionsformat för bättre zero/few-shot beteende.
**Continual Fine-Tuning** — Sekventiell finjustering på nya uppgifter utan att glömma tidigare.
**Reinforcement Fine-Tuning** — RL-baserad finjustering med reward signal efter SFT.
**Direct Preference Optimization** — DPO: optimerar preferenser direkt utan explicit reward model.
**Odds Ratio Preference Optimization** — ORPO: kombinerar SFT och preferensoptimering i ett steg.
**Kahneman-Tversky Optimization** — KTO: preferenslärande från binära good/bad labels utan parvis jämförelse.
**Supervised Fine-Tuning** — SFT: standard cross-entropy på demonstrationsdata före alignment.
**Curriculum Fine-Tuning** — Ordning av träningsdata från enkel till svår för stabilare konvergens.
**Data Selection for Fine-Tuning** — Urval av högkvalitativa/informativa exempel; minskar brus och kostnad.
**LIMA Hypothesis** — Hypotes att få högkvalitativa exempel kan räcka för stark instruction tuning.
**Overfitting in Fine-Tuning** — Modellen memorerar träningsinstruktioner och generaliserar dåligt till nya formuleringar.
**Early Stopping in Fine-Tuning** — Avbryter när valideringsmetrik försämras; förhindrar overfitting.
**Learning Rate for Fine-Tuning** — Typiskt 10–100× lägre än pretrain; för högt LR förstör basrepresentationer.
**Weight Decay in Fine-Tuning** — Regularisering som begränsar stora viktändringar från basmodellen.
**Gradient Accumulation in Fine-Tuning** — Simulerar större batch när GPU-minne begränsar batch size.
**Mixed Precision Fine-Tuning** — BF16/FP16 träning med FP32 master weights för hastighet.
**DeepSpeed Fine-Tuning** — Distribuerad finjustering med ZeRO och offload för stora modeller.
**FSDP Fine-Tuning** — Fully sharded finjustering som minskar per-GPU minneskrav.
**Multi-GPU Fine-Tuning** — Data parallel träning med synkroniserade gradienter över enheter.
**Single-GPU Fine-Tuning** — PEFT/QLoRA möjliggör finjustering av stora modeller på en konsument-GPU.
**Fine-Tuning Dataset Format** — JSONL med instruction/input/output eller messages-format enligt modell.
**ChatML Format** — Strukturerat meddelandeformat med role/content för konversationsfinjustering.
**Alpaca Format** — instruction/input/output tripplar populära i open-source finjustering.
**ShareGPT Format** — Konversationsloggar som träningsdata för chat-modeller.
**Packaged Fine-Tuning** — Packar korta exempel till fix sekvenslängd för effektiv träning.
**Truncation Strategy** — Hur långa exempel klipps; påverkar vilken information som bevaras.
**Max Sequence Length** — Övre gräns på tokens per träningsexempel; måste matcha modellkapacitet.
**Label Masking** — Maskerar förlust på prompt-delen så endast svar bidrar till gradient.
**Assistant-Only Loss** — Beräknar loss endast på assistant-svar i konversationer.
**NEFTune** — Lägger till brus i embeddings under träning; förbättrar ibland generalisering.
**Dropout During Fine-Tuning** — Regularisering; ofta lägre dropout än pretrain för att bevara kapacitet.
**Freezing Layers** — Fryser tidiga lager och finjusterar senare; vanligt i transfer learning.
**Gradual Unfreezing** — Frigör lager stegvis uppifrån och ned för stabil domain adaptation.
**Differential Learning Rates** — Olika LR per lagergrupp; lägre LR för tidiga lager.
**Task-Specific Head** — Separat output-lager per uppgift ovanpå delad encoder.
**Multi-Adapter Routing** — Väljer adapter per uppgift eller input vid inferens.
**Merge Adapters into Base** — Slår ihop tränade LoRA/adapter-vikter i basmodell för enklare deployment.
**LoRA Merge** — Beräknar W' = W + α/r·BA och sparar som en modell utan adapter-runtime.
**Fine-Tuning Evaluation** — Håll separat valideringsset som inte överlappar träningsinstruktioner.
**Held-Out Task Evaluation** — Testar på helt nya uppgiftsformat för att mäta generalisering.
**Benchmark Leakage in Fine-Tuning** — Risk att val/test-exempel hamnar i finjusteringsdata.
**Synthetic Fine-Tuning Data** — AI-genererade exempel för att skala domänspecifik finjustering.
**Distillation Fine-Tuning** — Student finjusteras mot teacher-logits eller svar på samma data.
**Self-Play Fine-Tuning** — Modell genererar och filtrerar egna träningsdata iterativt.
**Rejection Sampling Fine-Tuning** — Sample många svar, behåll bästa enligt verifierare, finjustera på dem.
**Iterative Fine-Tuning** — Upprepade finjusteringscykler med expanderande dataset.
**Fine-Tuning for Code** — Domänspecifik finjustering med kod-exempel, syntax och repo-kontext.
**Fine-Tuning for Math** — Träning på steg-för-steg-lösningar och verifierbara svar.
**Fine-Tuning for Tool Use** — Tränar modellen att välja verktyg och korrekt JSON-argument.
**Fine-Tuning for Safety** — Data och objektiv som minskar skadligt eller icke-efterlevande beteende.
**Negative Example Fine-Tuning** — Inkluderar avsiktligt dåliga svar märkta som fel för kontrast.
**Contrastive Fine-Tuning** — Optimerar representationer så positiva ligger nära och negativa långt.
**Unlikelihood Training** — Straffar sannolikhet för oönskade sekvenser under finjustering.
**Fine-Tuning Checkpoint Selection** — Väljer checkpoint med bäst validering, inte sista epoch.
**Hyperparameter Search for Fine-Tuning** — Söker LR, epoch, rank, batch size på valideringsmetrik.
**Epoch vs Step Budget** — Träningsbudget i pass genom data vs fix antal gradientsteg.
**Warmup in Fine-Tuning** — Kort LR-uppvärmning även vid finjustering för stabilitet.
**Cosine Decay Fine-Tuning** — LR-schema som mjuknar mot slutet av finjusteringskörning.
**Fine-Tuning Registry** — Versionering av dataset, hyperparametrar och checkpoints per experiment.
**Adapter Hub** — Bibliotek av delbara task-adapters som laddas ovanpå basmodell.
**Modular Fine-Tuning** — Separata moduler per färdighet som komponeras vid inferens.
**Fine-Tuning Drift** — Gradvis beteendeförändring vid upprepad finjustering utan basankare.
**Base Model Anchoring** — Regularisering mot ursprungliga vikter för att begränsa drift.

## Alignment & RLHF

**AI Alignment** — Design av AI-system vars mål och beteende stämmer med mänskliga värderingar och avsikter.
**Reinforcement Learning from Human Feedback** — RLHF: tränar policy med reward model tränad på mänskliga preferenser.
**Reward Model** — Modell som predicerar mänsklig preferens eller kvalitetspoäng för modelloutputs.
**Preference Dataset** — Par av svar där människor markerat vilket som är bättre; grund för alignment.
**Bradley-Terry Model** — Probabilistisk modell för parvisa preferenser: P(a>b) = σ(r(a)-r(b)).
**Pairwise Ranking Loss** — Förlust som uppmuntrar reward model att ranka preferred svar högre.
**Proximal Policy Optimization** — PPO: policy gradient-algoritm med klippad objektivfunktion för stabil RL.
**RLHF Pipeline** — Typiskt: SFT → reward model → PPO fine-tuning med KL-straff mot referenspolicy.
**KL Penalty to Reference** — Straffar avvikelse från SFT-policy för att undvika reward hacking och mode collapse.
**Reward Hacking** — Agent utnyttjar brister i reward model för högt score utan önskat beteende.
**Goodhart's Law in RLHF** — När reward blir mål förlorar den sin validitet som mått på verklig kvalitet.
**Constitutional AI** — Modell kritiserar och reviderar egna svar enligt definierade principer.
**RLAIF** — Reinforcement Learning from AI Feedback: AI istället för människor som preferensjudge.
**Direct Preference Optimization** — DPO: direkt preferensoptimering utan explicit RL-loop.
**Identity Preference Optimization** — IPO: variant av DPO med stabilare objektiv vid brusiga preferenser.
**SimPO** — Simple Preference Optimization: reference-free preferensmetod med sequence-level reward.
**Kahneman-Tversky Optimization** — KTO: optimerar från binära labels utan parvis jämförelse.
**ORPO** — Odds Ratio Preference Optimization: kombinerar SFT och preferenser i ett träningssteg.
**Rejection Sampling** — Generera många svar, filtrera med reward model, finjustera på bästa.
**Best-of-N Policy** — Väljer bästa av N samples enligt reward; ökar kvalitet utan viktuppdatering.
**Online RLHF** — Iterativ loop: samla data från aktuell policy → uppdatera reward → träna policy.
**Offline RLHF** — Tränar på statisk preferensdata utan interaktiv datainsamling.
**Human Preference Elicitation** — Metoder för att samla in jämförelser, rankningar eller skala-betyg.
**Annotator Disagreement** — Variation mellan mänskliga bedömningar; kräver aggregation och kvalitetskontroll.
**Inter-Rater Reliability** — Mått på konsistens mellan annotatörer (Cohen's kappa m.fl.).
**Red Teaming** — Systematisk adversarial testning för att hitta skadliga eller oönskade svar.
**Adversarial Prompting** — Prompts designade för att kringgå säkerhetsbegränsningar.
**Jailbreak** — Framgångsrik bypass av modellens säkerhetsinstruktioner.
**Refusal Behavior** — Modellens tränade avvisning av skadliga eller otillåtna förfrågningar.
**Over-Refusal** — Modellen avvisar legitima förfrågningar p.g.a. överkonservativ alignment.
**Helpfulness vs Harmlessness** — Tradeoff mellan att vara användbar och att undvika skada.
**Helpful Assistant Objective** — Optimerar för användarnytta inom säkerhetsgränser.
**Harmlessness Training** — Data och reward som straffar skadligt, olagligt eller vilseledande innehåll.
**Honesty Alignment** — Strävar efter sanningsenlighet och erkännande av osäkerhet.
**Sycophancy** — Tendens att hålla med användaren även när det är felaktigt.
**Value Alignment** — Modellens beteende överensstämmer med specificerade värderingar och normer.
**Normative Uncertainty** — Osäkerhet om vilka värderingar som ska optimeras i pluralistiska samhällen.
**Moral Uncertainty in AI** — Hur agenter ska agera när etiska principer konflikter.
**Scalable Oversight** — Metoder för att övervaka superhuman AI med begränsad mänsklig insats.
**Debate** — Två AI-agenter debatterar; människa bedömer vinnare för att extrahera sanning.
**Recursive Reward Modeling** — Människor bedömer AI som hjälper bedöma mer komplexa outputs.
**Iterated Amplification** — Upprepad distillation av mänsklig+AI-bedömning till mer kapabel agent.
**Process Supervision** — Belönar korrekta mellansteg i resonemang, inte bara slutgiltigt svar.
**Outcome Supervision** — Belönar endast korrekt slutresultat; enklare men svagare signal.
**Process Reward Model** — PRM betygsätter varje steg i kedja av tanke.
**Outcome Reward Model** — ORM betygsätter komplett svar.
**Verifiable Reward** — Automatiskt checkbar reward (t.ex. enhetstester för kod, matematisk verifiering).
**RL with Verifiable Rewards** — RLVR: RL där reward kommer från objektiv verifiering.
**Constitutional Principles** — Skriftliga regler som styr självrevision och träningsdata.
**Critique-Revision Loop** — Modell genererar kritik och förbättrat svar iterativt.
**Self-Critique** — Modellen utvärderar egna svar mot kriterier innan final output.
**Safety Fine-Tuning** — Finjustering specifikt på säkerhets- och policy-exempel.
**Toxicity Reduction** — Träning/reward som minskar hat, trakasserier och grovt språk.
**Bias Mitigation in Alignment** — Åtgärder för att minska stereotyper och unfair behandling.
**Preference Model Calibration** — RM:s scores ska korrelera med faktisk mänsklig preferens.
**Reward Model Overoptimization** — Policy utnyttjar RM:s blind spots när KL-straff är för svagt.
**Gold Standard Human Eval** — Högkvalitativa mänskliga bedömningar som referens för automatiska metriker.
**LLM-as-Judge** — Använder LLM för att bedöma svar; skalbart men med egna bias.
**Judge Prompt Design** — Instruktioner till judge-LLM som påverkar bedömningens validitet.
**Position Bias in LLM Judges** — Judge favoriserar svar i viss position i parvis jämförelse.
**Length Bias in Reward** — RM eller judge favoriserar längre svar oavsett kvalitet.
**Style Bias in Preferences** — Preferenser driven av ton/stil snarare än korrekthet.
**Multi-Objective Alignment** — Balanserar flera mål (helpful, harmless, honest) samtidigt.
**Pareto Frontier of Alignment** — Mängd av policyer där ingen måldimension kan förbättras utan att försämra annan.
**Constraint Optimization in RLHF** — Maximera reward under hårda säkerhetsconstraints.
**Lagrangian RLHF** — Dynamiska multiplikatorer balanserar reward vs KL/safety-straff.
**Safe RLHF** — Integrerar explicit säkerhetsmodell i RLHF-optimering.
**Toxicity Classifier Guardrail** — Filter som blockerar eller omskriver toxisk output post-hoc.
**Moderation API** — Extern klassificerare för policyöverträdelser i input/output.
**Content Policy** — Regler för tillåtet/innehåll som styr träning och deployment.
**Alignment Tax** — Prestationsförlust på vissa uppgifter efter säkerhets-/preferensoptimering.
**Capability-Alignment Tradeoff** — Starkare capabilities kan öka risk om alignment inte skalar.
**Deceptive Alignment** — Teoretisk risk: modell appearar aligned under träning men inte i deployment.
**Inner Alignment** — Agentens interna mål matchar avsedda träningsmål.
**Outer Alignment** — Specificerade träningsmål matchar designerens intentioner.
**Mesaa Optimization** — Optimering av proxy-mått som divergerar från avsedda mål.
**Specification Gaming** — Agent uppfyller bokstavlig specifikation men inte avsedd intent.
**Inverse Reinforcement Learning** — IRL: infererar reward function från demonstrations.
**Cooperative Inverse RL** — Människa och agent samarbetar; agent infererar mänskliga mål.
**Human-in-the-Loop RL** — Människa ger feedback under pågående policy learning.
**Active Preference Learning** — Väljer de jämförelser som ger mest information om preferenser.
**Distributional Shift in RLHF** — Policy-genererad data skiljer sig från SFT-data; RM kan extrapolera dåligt.
**On-Policy vs Off-Policy RLHF** — On-policy: tränar på data från aktuell policy; off-policy: statisk data.
**Reference Model in DPO** — Fryst SFT-modell som implicit KL-ankare i DPO-objektivet.
**Beta in DPO** — Temperaturparameter som styr styrka av preferensoptimering vs KL till referens.
**Alignment Evaluation Suite** — Samling av säkerhets-, sanning- och hjälpsamhetstester.
**Harm Benchmark** — Standardiserade prompts för att mäta skadlig modelloutput.
**TruthfulQA Alignment** — Benchmark för sanningsenlighet vs populära missuppfattningar.
**MT-Bench Human Alignment** — Multi-turn konversationskvalitet bedömd av människor eller judge.
**Reward Model Ensemble** — Flera RM kombineras för robustare reward signal.

## RAG & Retrieval

**Retrieval-Augmented Generation** — RAG: hämtar externa dokument och conditionar generering på dem för aktuell kunskap.
**Vector Database** — Databas optimerad för likhetssökning i högdimensionella embeddings.
**Embedding Model** — Modell som mappar text till dense vektorer för semantisk likhetssökning.
**Dense Retrieval** — Hämtning via embedding-likhet istället för lexikal keyword-matchning.
**Sparse Retrieval** — Klassisk IR med termvikter (BM25, TF-IDF) utan neural embeddings.
**Hybrid Retrieval** — Kombinerar sparse och dense scores för robustare recall.
**BM25** — Probabilistisk rankingfunktion baserad på termfrekvens och dokumentlängdsnormalisering.
**TF-IDF** — Term frequency-inverse document frequency; enkel viktning av viktiga termer.
**Approximate Nearest Neighbor Search** — ANN: snabb ungefärlig sökning i stora vektorindex (HNSW, IVF).
**HNSW Index** — Hierarchical Navigable Small World: graf-baserat ANN-index med hög recall.
**IVF Index** — Inverted File Index: klustrar vektorer och söker i relevanta kluster.
**Product Quantization** — PQ: komprimerar vektorer för minnes- och sök-effektivitet med liten precisionförlust.
**Cosine Similarity** — Mått på vinkel mellan vektorer; standard för normaliserade embeddings.
**Dot Product Retrieval** — Inner product som likhet; kräver ofta normaliserade embeddings.
**Euclidean Distance in Retrieval** — L2-avstånd som alternativ likhetsmetrik i vektorrum.
**Chunking Strategy** — Delar dokument i segment för indexering; påverkar recall och precision.
**Fixed-Size Chunking** — Delar text i fix token-/teckenlängd med eventuell overlap.
**Semantic Chunking** — Delar vid naturliga avsnittsgränser baserat på struktur eller embeddings.
**Parent-Document Retriever** — Hämtar liten chunk men returnerar större omgivande kontext.
**Sliding Window Chunking** — Överlappande fönster för att undvika att information klipps vid gränser.
**Document Ingestion Pipeline** — ETL: ladda, rensa, chunka, embedda och indexera källor.
**Metadata Filtering** — Filtrerar retrieval på metadata (datum, källa, taggar) före likhetssökning.
**Multi-Vector Retrieval** — Flera embeddings per dokument (ColBERT-style) för finare matchning.
**ColBERT** — Late interaction: token-nivå embeddings med MaxSim-scoring vid retrieval.
**Cross-Encoder Reranker** — Joint encoding av query+doc för exakt relevance score; dyr men precis.
**Bi-Encoder Retrieval** — Separata encoders för query och doc; snabb ANN-sökning.
**Reranking Stage** — Andra pass som sorterar top-k kandidater med kraftfullare modell.
**Retrieve-then-Read** — Klassisk RAG: hämta dokument → mata in i LLM som kontext.
**Retrieve-then-Generate** — Generering conditionad på hämtade passager utan explicit reader-modell.
**Query Rewriting** — Omformulerar användarfråga för bättre retrieval (HyDE, step-back).
**HyDE** — Hypothetical Document Embeddings: genererar hypotetiskt svar och söker med dess embedding.
**Multi-Query Retrieval** — Genererar flera query-varianter och union av resultat.
**Step-Back Prompting** — Genererar mer abstrakt fråga parallellt för bredare retrieval.
**Self-RAG** — Modellen beslutar när den ska hämta, relevansbedöma och kritisera egna svar.
**Corrective RAG** — CRAG: verifierar retrieval-kvalitet och kan söka om eller använda web fallback.
**Adaptive RAG** — Väljer retrieval-strategi baserat på frågetyp och konfidens.
**GraphRAG** — Bygger kunskapsgraf från korpus för strukturerad community-baserad retrieval.
**Agentic RAG** — Agent planerar flera retrieval-steg och verktygsanrop dynamiskt.
**Multi-Hop Retrieval** — Flera sekventiella sökningar för att samla bevis över dokument.
**Fusion-in-Decoder** — FiD: encoder processar flera docs separat; decoder fuserar dem.
**Context Window Budget** — Max tokens för hämtade docs + prompt; kräver prioritering/truncation.
**Lost in the Middle** — LLM ignorerar information i mitten av lång kontext; påverkar RAG-layout.
**Context Ordering** — Placering av viktigaste docs först/sist kan förbättra att de används.
**Citation in RAG** — Modellen refererar källor; kräver träning eller post-processing.
**Attribution Evaluation** — Mäter om genererade påståenden stöds av hämtade källor.
**Faithfulness in RAG** — Output följer hämtad evidens utan fabricerade tillägg.
**Answerability Detection** — Bedömer om hämtad kontext räcker för att besvara frågan.
**Abstention in RAG** — Modellen avstår svara när evidens är otillräcklig.
**Hallucination in RAG** — Modellen fabricerar trots tillgänglig kontext; kvarstående risk.
**Grounding** — Binda svar till specifik källtext eller fakta.
**Knowledge Cutoff** — Datum efter vilket basmodell saknar kunskap; RAG kompenserar.
**Freshness in Retrieval** — Prioriterar nyare dokument vid tidskänsliga frågor.
**Incremental Index Update** — Lägger till/uppdaterar vektorer utan full reindex.
**Embedding Drift** — Förändrad embedding-modell gör gamla index inkompatibla.
**Reindexing** — Full ombyggnad av vektorindex efter modell- eller schemaändring.
**Deduplication in Corpus** — Tar bort duplicerade/near-duplicate chunks före indexering.
**PII Redaction in Ingestion** — Maskerar persondata innan indexering och retrieval.
**Access Control in RAG** — Filtrerar retrieval per användares behörigheter.
**Multi-Tenant RAG** — Isolerade index per kund med delad infrastruktur.
**Retrieval Recall@k** — Andel queries där relevant doc finns i top-k.
**Retrieval MRR** — Mean Reciprocal Rank: belönar högt rankad första relevanta träff.
**Retrieval NDCG** — Normaliserad discounted cumulative gain; rankningskvalitet med graded relevance.
**Hit Rate** — Andel queries med minst en relevant träff i top-k.
**RAGAS Framework** — Automatiska metriker: faithfulness, answer relevance, context precision/recall.
**Context Precision** — Andel hämtade chunks som faktiskt är relevanta för frågan.
**Context Recall** — Andel nödvändig information som hämtades från källan.
**Answer Relevance** — Genererat svar adresserar frågan utan irrelevant innehåll.
**Noise Robustness in RAG** — Prestanda när retrieval inkluderar irrelevanta dokument.
**Negative Retrieval** — Medvetet irrelevanta docs i kontext; testar modellens filtrering.
**Long-Context RAG** — Hämtar många eller långa docs när LLM har stor kontext.
**Small-to-Big Retrieval** — Hämta små chunks, expandera till större kontext vid generering.
**Sentence Window Retrieval** — Hämtar mening med omgivande fönster som kontext.
**Auto-Merging Retriever** — Slår ihop relaterade chunks hierarkiskt efter retrieval.
**Knowledge Graph Retrieval** — Söker i KG noder/kanter och serialiserar till LLM-kontext.
**Structured Data RAG** — SQL/API över tabeller kombinerat med vektor-sökning.
**Tool-Augmented RAG** — Agent anropar search/API/SQL som retrieval-verktyg.
**Web Search RAG** — Live webbsökning som dynamisk kunskapskälla.
**Cache-Augmented Generation** — CAG: förcomputad kunskap i KV-cache istället för runtime retrieval.
**Prompt Compression for RAG** — Komprimerar hämtad kontext (LLMLingua) för att passa budget.
**Contextual Chunk Headers** — Prepender metadata/rubrik till chunk för bättre disambiguation.
**Late Chunking** — Embedda hel dokument först, chunka i embedding-space.
**Matryoshka Embeddings** — Truncerbara embeddings för flexibel precision/latens-tradeoff.
**Embedding Fine-Tuning for Domain** — Domänspecifik finjustering av retriever för bättre recall.
**Hard Negative Mining** — Tränar retriever med svåra felaktiga docs som negativa exempel.
**Contrastive Retrieval Training** — InfoNCE-lik förlust som drar query nära relevant doc.
**In-Batch Negatives** — Andra docs i batch används som negativa i contrastive träning.
**Query-Document Asymmetry** — Query och doc kan ha olika encoders eller instruktioner.
**Instruction-Tuned Retriever** — Retriever tränad med instruktioner per uppgiftstyp.
**Multi-Lingual Retrieval** — Cross-lingual sökning: fråga på ett språk, docs på annat.
**RAG Latency Budget** — Total tid för embed + search + rerank + generate.

## AI-agenter

**AI Agent** — Autonom LLM-baserad system som planerar, använder verktyg och agerar mot mål i flera steg.
**Agent Loop** — Observe → plan → act → observe cykel tills uppgift är klar eller budget slut.
**ReAct Pattern** — Reasoning + Acting: modellen alternerar tanke och verktygsanrop i samma trace.
**Plan-and-Execute** — Separerar planering från exekvering; planner skapar steg som executor följer.
**Tool Calling** — Strukturerat API där modellen väljer verktyg och JSON-argument.
**Function Schema** — JSON Schema som beskriver verktygets namn, parametrar och typer för LLM.
**Tool Registry** — Katalog över tillgängliga verktyg med metadata och access policies.
**Agent Memory** — Kort- och långtidsminne för att bevara kontext, fakta och tidigare erfarenheter.
**Short-Term Agent Memory** — Aktuell konversation och scratchpad inom kontextfönster.
**Long-Term Agent Memory** — Persistent lagring (vektor-DB, profil) utöver en session.
**Episodic Memory** — Lagrar specifika händelser/interaktioner som referens vid framtida uppgifter.
**Semantic Memory** — Generaliserade fakta och kunskap extraherade från erfarenheter.
**Working Memory Scratchpad** — Explicit yta där agenten skriver mellanresultat och delplaner.
**Multi-Agent System** — Flera specialiserade agenter samarbetar via meddelanden eller delad state.
**Orchestrator Agent** — Koordinerar sub-agenter, delegerar deluppgifter och aggregerar resultat.
**Hierarchical Agents** — Manager-agenter delegerar till worker-agenter i trädstruktur.
**Agent Communication Protocol** — Standardiserat meddelandeformat mellan agenter (t.ex. MCP, A2A).
**Model Context Protocol** — MCP: öppet protokoll för att koppla LLM till verktyg och datakällor.
**Agent-to-Agent Protocol** — A2A: interoperabilitet mellan agenter från olika leverantörer.
**Handoff Between Agents** — Överlämnande av konversation/uppgift från en agent till en annan.
**Subagent Delegation** — Parent-agent spawnar child med begränsat scope och verktyg.
**Agent State Machine** — Explicita tillstånd (planning, executing, waiting) styr agentflöde.
**Goal Decomposition** — Bryter högnivåmål i delmål och atomära actions.
**Task Planning** — Genererar ordnad lista av steg för att uppnå mål.
**Dynamic Replanning** — Uppdaterar plan när verktygsresultat avviker från förväntan.
**Reflection in Agents** — Agent utvärderar egna steg och korrigerar fel (Reflexion, self-refine).
**Self-Refinement Loop** — Generera → kritik → förbättra iterativt utan extern reward.
**Verifier Agent** — Separat agent/modell som kontrollerar korrekthet före final svar.
**Critic Agent** — Bedömer kvalitet på plan eller output och ger korrigerande feedback.
**Code Interpreter Agent** — Kör Python/sandbox för beräkning, analys och filhantering.
**Browser Agent** — Navigerar webben via browser automation för information eller actions.
**Computer Use Agent** — Styr GUI (mus/tangentbord) för att interagera med applikationer.
**Sandboxed Execution** — Isolerad miljö för agentkod med begränsade permissions.
**Tool Timeout** — Max tid per verktygsanrop för att undvika hängande agenter.
**Max Iteration Limit** — Övre gräns på agent-loopar; förhindrar oändliga loopar.
**Action Budget** — Max antal verktygsanrop eller tokens per uppgift.
**Human-in-the-Loop Agent** — Människa godkänner kritiska actions innan exekvering.
**Approval Gate** — Policy som kräver explicit bekräftelse för känsliga operationer.
**Autonomous vs Semi-Autonomous** — Full autonomi vs mänsklig övervakning vid viktiga beslut.
**Agent Observability** — Logging av thoughts, tool calls, latens och fel för debugging.
**Trace Visualization** — UI som visar agentens resonemang och verktygskedja.
**OpenTelemetry for Agents** — Standardiserad tracing av agent-spans och verktygsanrop.
**Agent Evaluation** — Mäter task success rate, steg-effektivitet och säkerhet.
**SWE-Bench Agent** — Benchmark för agenter som löser verkliga GitHub issues med kod.
**WebArena Benchmark** — Simulerade webbuppgifter för att utvärdera web-agenter.
**Tool Use Accuracy** — Andel korrekt formaterade och semantiskt rätta verktygsanrop.
**Planning Error Recovery** — Agentens förmåga att återhämta sig från felaktiga plansteg.
**Ambiguity Handling** — Agenten ställer förtydligande frågor vid underspecificerade mål.
**Multi-Modal Agent** — Agent som hanterar text, bild, ljud och filer via verktyg.
**Retrieval Agent** — Specialiserad agent för sökning och sammanställning av information.
**Research Agent** — Iterativ informationssökning, syntes och källkritik.
**Coding Agent** — Skriver, kör tester och itererar kod i repo-miljö.
**DevOps Agent** — Automatiserar CI/CD, infra och incident response.
**Customer Support Agent** — Hanterar ärenden med CRM/KB-integration och eskalering.
**Personal Assistant Agent** — Kalender, e-post och påminnelser med användarpreferenser.
**Agent Persona** — Konfigurerad roll, ton och capabilities som styr beteende.
**System Prompt for Agents** — Instruktioner om verktyg, säkerhet och planeringsformat.
**Structured Action Format** — XML/JSON/YAML för actions som parsas deterministiskt.
**Parallel Tool Calls** — Flera oberoende verktyg körs samtidigt för lägre latens.
**Sequential Tool Dependency** — Senare steg kräver output från tidigare verktyg.
**Error Propagation in Agents** — Hur fel i verktyg rapporteras tillbaka till planner.
**Retry Policy** — Regler för omförsök vid transienta verktygsfel.
**Idempotent Tool Design** — Verktyg säkra att köra flera gånger utan sidoeffekter.
**Side-Effect Tracking** — Loggar muterande actions för audit och rollback.
**Agent Rollback** — Ångra felaktiga ändringar efter misslyckad verifiering.
**Checkpoint in Agent Run** — Sparar state för att återuppta långa uppgifter.
**Streaming Agent Output** — Streamar delvis plan och resultat till användaren i realtid.
**Background Agent** — Kör asynkront medan användaren gör annat; notifierar vid klart.
**Scheduled Agent Task** — Cron-liknande körning av agent på schema.
**Event-Triggered Agent** — Startar vid webhook/event (nytt ärende, alert).
**Multi-Turn Tool Use** — Verktygsanrop över flera konversationsvarv med state.
**Context Overflow in Agents** — Komprimering/summary när historik överstiger kontext.
**Conversation Summarization** — Periodisk sammanfattning av tidigare steg för att spara tokens.
**Agent Prompt Injection Defense** — Filtrerar otillåten styrning från verktygsoutput/extern data.
**Privilege Separation** — Olika verktyg/permissions per agent-roll.
**Least Privilege for Tools** — Minimal API-access per uppgift för att begränsa skada.
**Agent Guardrails** — Policylager som blockerar otillåtna actions före exekvering.
**Deterministic Tool Routing** — Regelbaserad routing istället för LLM-val vid kritiska paths.
**LLM Router Agent** — Liten modell klassificerar intent och väljer specialist-agent.
**Mixture of Agents** — Flera agenter genererar svar; aggregator kombinerar bästa.
**Debate Between Agents** — Agenter argumenterar mot varandra; domare väljer svar.
**Consensus Agent Protocol** — Kräver överenskommelse mellan agenter före action.
**Agent Framework** — Ramverk (LangGraph, AutoGen) för att bygga agentflöden.
**LangGraph** — Graf-baserat orchestration för stateful LLM-agenter med cykler.
**AutoGen** — Multi-agent konversationsramverk med programmerbara agenter.
**CrewAI** — Roll-baserade agenter i team med delegerad arbetsfördelning.
**Agent SDK** — Bibliotek för tool definitions, runs och streaming (OpenAI Agents SDK).
**Computer-Using Agent Safety** — Risker när agent styr riktiga system; kräver sandboxes och limits.
**Agent Cost Control** — Token-, API- och tidsbudget per run.
**Success Criteria Specification** — Maskinläsbara kriterier för när agenten ska stoppa.

## Multimodal AI

**Multimodal Model** — Modell som processar och genererar flera modaliteter (text, bild, ljud, video).
**Modality Alignment** — Träning så att representationer från olika modaliteter ligger i delat semantiskt rum.
**Vision-Language Model** — VLM: joint modellering av bild och text för förståelse och generering.
**Image Encoder** — Neural encoder (CNN/ViT) som mappar bild till vektorrepresentation.
**Text Encoder in VLM** — Transformer-encoder för text som projiceras till samma rum som bildfeatures.
**Projection Layer** — Linjärt/MLP-lager som mappar en modalitet till gemensam embedding-dimension.
**Contrastive Language-Image Pretraining** — CLIP: tränar bild- och textencoder med contrastive loss på par.
**CLIP Embedding Space** — Delat vektorrum där matchande bild-text-par ligger nära varandra.
**Zero-Shot Image Classification** — Klassificering via textprompter utan task-specifik träning, med CLIP-liknande modeller.
**Image-Text Matching** — Bedömer om bild och text beskriver samma innehåll.
**Visual Question Answering** — VQA: svara på naturliga språkfrågor om bildinnehåll.
**Image Captioning** — Generera textbeskrivning av bild; encoder-decoder eller VLM-generering.
**Text-to-Image Generation** — Generera bild från textprompt via diffusion eller autoregressiva modeller.
**Interleaved Image-Text Training** — Träning på sekvenser som växlar bild- och texttokens.
**Any-to-Any Multimodal** — Modell som mappar godtycklig input-modalitet till godtycklig output.
**Unified Tokenization** — Representerar bild/ljud/video som token-sekvenser i samma vocab/embedding.
**Visual Tokenizer** — VQ-VAE eller patch-embedding som diskretiserar bild till tokens.
**Audio Tokenizer** — Neural codec (EnCodec) som kodar ljud till diskreta tokens.
**Video Frame Encoding** — Samplear och encoderar bildrutor som sekvens eller 3D-volym.
**Temporal Modeling in Video** — Fångar tidssamband via 3D-conv, attention över frames eller state space.
**Speech-to-Text** — ASR: transkriberar tal till text med encoder-decoder eller CTC.
**Text-to-Speech** — TTS: syntetiserar naturligt tal från text.
**Speech Encoder** — Modell (wav2vec, Whisper encoder) som extraherar fonetiska/semantiska features.
**Whisper Architecture** — Encoder-decoder ASR tränad på massiv weakly supervised ljuddata.
**Audio-Language Model** — LLM som tar ljudtokens eller embeddings som input för förståelse/generering.
**Omni Model** — En modell som hanterar text, bild, ljud in/ut i en pipeline.
**Cross-Modal Attention** — Attention mellan tokens från olika modaliteter i samma sekvens.
**Early Fusion** — Kombinerar modaliteter i tidiga lager av nätverket.
**Late Fusion** — Separata encoders; fusion i slutet före task head.
**Middle Fusion** — Korsmodal interaktion i mellanliggande lager.
**Perceiver Multimodal** — Latent bottleneck attendar över flera modaliteter oberoende av storlek.
**Flamingo Architecture** — Fryst LLM med cross-attention till bildfeatures från vision encoder.
**LLaVA Architecture** — Projicerar CLIP/ViT-features till LLM via MLP connector.
**Q-Former** — Query transformer som extraherar fix antal visuella tokens till LLM.
**AnyRes Resolution Handling** — Dela högupplösta bilder i variabla patch-grid för VLM.
**Dynamic Resolution Input** — Variabel bildstorlek utan fix resize; bättre detaljbevarande.
**OCR in VLM** — Optisk teckenläsning integrerad i vision-language för dokument.
**Document Understanding** — Layout, tabeller och text i PDF/bild via multimodal modeller.
**Chart and Diagram Reasoning** — Tolkning av grafer, diagram och visualiseringar i VLM.
**Spatial Reasoning in Vision** — Förståelse av position, storlek och relationer i bilder.
**Grounding via Bounding Boxes** — Kopplar språk till regioner med koordinater eller masker.
**Referring Expression Comprehension** — Hitta objekt i bild givet naturlig språkbeskrivning.
**Segmentation from Text Prompt** — Text-styrd segmentering (SAM + CLIP eller unified models).
**Multimodal In-Context Learning** — Few-shot med bild+text-exempel i prompten.
**Interleaved Generation** — Generera text och bild växelvis i samma session.
**Image Editing via Language** — Modifiera bild med instruktioner (inpainting, style, object swap).
**Video Captioning** — Beskriv video-innehåll i naturligt språk.
**Video QA** — Frågesvar om händelser och objekt i video.
**Multimodal RAG** — Hämtar bilder, diagram och text till gemensam kontext.
**Multimodal Embedding Search** — Sök bilder med text eller vice versa i delat index.
**Modality Missing at Inference** — Hantera saknad modalitet (endast text eller endast bild).
**Modality Dropout** — Slumpmässigt droppar modalitet under träning för robusthet.
**Alignment Loss** — Contrastive eller matching loss som synkar modalitetsrepresentationer.
**ITC Loss** — Image-Text Contrastive loss i CLIP-liknande träning.
**ITM Loss** — Image-Text Matching binary classification loss.
**LM Loss on Captions** — Autoregressiv caption loss som komplement till contrastive.
**Multimodal Pretraining Data** — Web-scale bild-text-par, videos, interleaved web documents.
**Data Filtering for Multimodal** — Kvalitetsfilter för NSFW, blur, mismatch mellan bild och alt-text.
**Synthetic Multimodal Data** — Renderade scener, captions från LLM för träning.
**Audio-Visual Learning** — Joint modellering av ljud och video (t.ex. lip sync, AVSR).
**Lip Reading Model** — Predicerar tal från munrörelser i video utan ljud.
**Music Generation Multimodal** — Generera musik från text eller humör-beskrivning.
**3D Understanding** — Point clouds, meshes och NeRF som input till AI-modeller.
**Point Cloud Encoder** — Network på 3D-punkter (PointNet, transformers on patches).
**NeRF Representation** — Neural Radiance Fields: implicit 3D-scen som kan renderas från vyer.
**Gaussian Splatting** — Explicit 3D Gaussians för realtid rendering; används i 3D-gen AI.
**Text-to-3D** — Generera 3D-objekt/scener från text via diffusion eller optimization.
**Multimodal Safety** — Filter för skadligt bild/text-innehåll och deepfake-risk.
**Deepfake Detection** — Klassificera AI-genererade ansikten/röster vs autentiska.
**Provenance Metadata** — C2PA m.m. som spårar ursprung av genererat media.
**Multimodal Benchmark** — Evalueringssuite för VQA, captioning, grounding (MME, MMMU).
**MMMU Benchmark** — Massive Multi-discipline Multimodal Understanding; expertfrågor med bilder.
**HallusionBench** — Testar hallucination i VLM på visuella påståenden.
**Multimodal CoT** — Chain-of-thought med visuella mellansteg eller region reasoning.
**Set-of-Mark Prompting** — Overlay numrerade markörer på bild för att guida VLM-attention.
**Image Token Budget** — Max antal visuella tokens; påverkar detalj vs latens.
**Video Token Budget** — Subsampling frames eller temporal compression för långa videos.
**Modality Adapter** — Lättviktsmodul som kopplar ny modalitet till fryst LLM.
**Unified Multimodal Decoder** — En decoder genererar tokens för alla modaliteter.
**Dual Encoder Retrieval** — Separata encoders per modalitet med shared contrastive space.
**Cross-Modal Retrieval** — Hämta bild givet text eller text givet bild.
**Multimodal Fine-Tuning** — Finjustera VLM på domänspecifika bild-text-uppgifter.
**Instruction Tuning for VLM** — Multimodal chat/instruktionsdata för dialog om bilder.
**Negative Image-Text Pairs** — Hard negatives i contrastive träning för skärpare gränser.
**Resolution Extrapolation** — Inferens på högre upplösning än träningsdata.
**Aspect Ratio Handling** — Bevarar bildproportioner vid patchificering.
**Color Space in Vision Models** — RGB vs YUV; normalisering påverkar pretrained features.
**Multimodal Latency** — Vision encoder + LLM decode; ofta flaskhals i encoder.
**Visual Grounding** — Kopplar naturligt språk till specifika bildregioner via boxar, masker eller pekning.
**Audio-Visual Speech Recognition** — Kombinerar lip-read video och ljud för robustare taligenkänning i brus.

## Diffusionsmodeller

**Diffusion Model** — Generativ modell som lär sig reversera gradvis brusning av data till sampling.
**Forward Diffusion Process** — Markovkedja som adderar Gaussisk brus till data över T tidssteg.
**Reverse Diffusion Process** — Lär parametriserad modell att stegvis avlägsna brus och återskapa data.
**Noise Schedule** — Varians β_t över tidssteg; styr hur snabbt signal försvinner.
**DDPM** — Denoising Diffusion Probabilistic Models: grundläggande diskret tidssteg-diffusion.
**DDIM** — Deterministisk sampler med färre steg än DDPM utan extra träning.
**Score Matching** — Lär ∇_x log p(x) (score function) istället för explicit densitet.
**Score-Based Generative Model** — SGM: SDE/ODE-perspektiv på diffusion och sampling.
**Denoising Score Matching** — Träna nätverk att predicera brus/skore givet noised input.
**Epsilon Prediction** — Modellen predicerar tillagt brus ε istället för x_0 direkt.
**x0 Prediction** — Modellen predicerar ren data x_0 från noised sample.
**v-Prediction** — Interpolerad prediktionstarget som stabiliserar träning vid varierande SNR.
**Signal-to-Noise Ratio** — SNR: förhållande signal/brus vid tid t; styr loss-viktning.
**Variance Preserving Schedule** — VP-SDE: bevarar varians ungefär konstant under forward process.
**Variance Exploding Schedule** — VE-SDE: varians växer; alternativ brusningsformulering.
**Latent Diffusion Model** — LDM: diffusion i komprimerat latent rum från VAE/autoencoder.
**VAE in Latent Diffusion** — Autoencoder komprimerar bild till latent z där diffusion sker.
**U-Net Denoiser** — U-Net-arkitektur som predicerar brus/skore med skip connections.
**Cross-Attention Conditioning** — Text/embeddings conditionar U-Net via cross-attention (Stable Diffusion).
**Classifier-Free Guidance** — CFG: kombinerar conditional och unconditional prediktion för skarpare samples.
**Guidance Scale** — Vikt på CFG-term; högre → starkare prompt-följsamhet men kan over-saturate.
**Text Encoder for Diffusion** — CLIP/T5 encoder som producerar text embeddings till U-Net.
**Timestep Embedding** — Sinusoidal eller learned embedding av diffusionsteg t till nätverket.
**Class-Conditional Diffusion** — Conditionar på klasslabel via embedding eller adm.
**Inpainting Diffusion** — Maskerad region fylls i medan ok-maskerade områden conditionar.
**Outpainting** — Genererar utökning utanför originalbildens kanter.
**Image-to-Image Diffusion** — Startar från noised version av källbild med strength-parameter.
**ControlNet** — Auxiliary network injicerar spatial control (edges, depth, pose) i U-Net.
**T2I-Adapter** — Lättvikts adapter för strukturell kontroll utan full ControlNet.
**LoRA for Diffusion** — Lågranks finjustering av U-Net/text encoder för stilar/koncept.
**DreamBooth** — Finjustering på få bilder för att lära specifikt subjekt/koncept.
**Textual Inversion** — Lär nytt 'word' embedding i text encoder från få exempel.
**IP-Adapter** — Image prompt adapter: conditionar generering på referensbild.
**Regional Prompting** — Olika textprompter för olika spatiala regioner.
**Negative Prompt** — Text som modellen ska undvika via CFG unconditional branch.
**Sampler** — Algoritm som integrerar reverse process (Euler, DPM++, Heun).
**Euler Discrete Sampler** — Enkel ODE-lösare med få steg; populär i Stable Diffusion.
**DPM-Solver** — Högre ordningens ODE-lösare för snabb sampling med få steg.
**UniPC Sampler** — Unified predictor-corrector för effektiv diffusion sampling.
**SDE vs ODE Sampling** — SDE ger stokastisk variation; ODE mer deterministisk.
**Rectified Flow** — Lär direkt transport mellan noise och data längs raka paths.
**Flow Matching** — Kontinuerlig normaliseringsflödes-träning utan simulering av full diffusion.
**Consistency Model** — En-stegs eller få-stegs generator tränad för konsistent denoising.
**Distillation for Diffusion** — Student modell lär sig få-stegs sampling från teacher.
**Progressive Distillation** — Iterativt halverar antal sampling-steg via distillation.
**Turbo/LCM Models** — Latent Consistency Models för realtid få-stegs generering.
**Video Diffusion** — 3D U-Net eller temporal attention för video-generering.
**Temporal Attention in Video Diffusion** — Attention över frames för tidskoherens.
**Audio Diffusion** — Diffusion i spektrogram eller latent ljudrepresentation.
**Diffusion Transformer** — DiT: transformer istället för U-Net som denoiser backbone.
**Patchified Latent Input** — Delar latent bild i patchar som tokens till DiT.
**EDM Framework** — Elucidating Diffusion Models: unified formulering av schedules och loss.
**Per-Resolution Training** — Multi-scale träning för bättre detalj och stabilitet.
**Min-SNR Weighting** — Viktning av loss baserat på SNR för balanserad träning.
**EMA Weights** — Exponential moving average av modellvikter för stabilare sampling.
**Diffusion Training Steps** — Antal tidssteg T under träning; kan skilja från inference steg.
**Inference Step Count** — Färre steg än träning via avancerade samplers; tradeoff kvalitet/hastighet.
**Mode Collapse in Diffusion** — Sällsynt men möjlig via dålig guidance eller data bias.
**Exposure Bias in Diffusion** — Mindre relevant än autoregressiv; men schedule/sampler påverkar artefakter.
**Safety Filter for Diffusion** — NSFW-klassificerare på prompt/output.
**Watermarking Generated Images** — Osynlig vattenmärkning i diffusion outputs.
**Causal Diffusion for Video** — Kausala temporal constraints för streaming video-gen.
**Conditional Dropout** — Slumpmässigt droppar conditioning under träning för CFG-kompatibilitet.
**Null Text Embedding** — Unconditional embedding för CFG unconditional branch.
**Prompt Weight Syntax** — Syntax (emphasis) för att vikta delar av prompt i inference.
**CLIP Score for Evaluation** — Cosine similarity CLIP(image, text) som kvalitetsproxy.
**FID for Diffusion** — Fréchet Inception Distance mellan genererade och riktiga bildfeatures.
**IS Inception Score** — Mått på bildkvalitet och diversitet via classifier entropy.
**Human Preference for T2I** — Elo/ranking av bilder baserat på estetik och prompt-match.
**GenEval Benchmark** — Objektantal, färg och spatial relation i T2I-evaluering.
**DPMSolver++** — Förbättrad DPM solver med bättre stabilitet vid låga steg.
**Karras Sigmas** — Noise schedule formulering optimerad för få-stegs sampling.
**Sigma Schedule** — Explicit brusnivå σ(t) istället för β_t i vissa implementationer.
**Diffusion Model Serving** — Batchad inferens med shared text encoding och VAE decode.
**VAE Decoder Artifacts** — Blur/ringing från latent decode; påverkar perceived skärpa.
**Tiled VAE Decode** — Decode stora bilder i tiles för att passa GPU-minne.
**MultiDiffusion** — Generera stora bilder genom överlappande diffusion-fönster.
**Semantic Diffusion Guidance** — Extra guidance från semantic segmentation/CLIP gradients.
**Prompt-to-Prompt** — Redigera bild genom att manipulera cross-attention maps.
**Attention Store in Diffusion** — Sparar attention maps för editing och interpretability.
**Null-Label Training** — Tränar unconditional branch parallellt med conditional.
**Diffusion Prior** — Separat diffusion över embeddings (DALL-E 2 prior) före decoder.
**Cascaded Super-Resolution** — Lågupplöst diffusion + upsampler för högupplöst output.
**Imagen Architecture** — Kaskad text-to-image med frozen T5 och super-res stages.
**Noise Offset Training** — Adderar liten konstant offset till brus för bättre kontrast/ljus.
**Offset Noise** — Low-frequency bruskomponent i träning för ljusvariation.
**Diffusion Model Quantization** — INT8/INT4 U-Net för snabbare inferens med minimal kvalitetsförlust.
**Diffusion Model Compilation** — torch.compile/onnx för optimerad inference pipeline.
**Stochastic Sampler** — Sampling som injicerar brus vid varje steg; ger mer variation än ren ODE-lösning.
**Deterministic Sampler** — Integrerar reverse ODE utan extra brus; reproducerbar given startseed.

## Computer Vision

**Convolutional Neural Network** — CNN: hierarkiska filter som fångar lokala spatiala mönster i bilder.
**Vision Transformer** — ViT: behandlar bildpatchar som tokens med transformer-encoder.
**Image Classification** — Tilldelar hel bild en klasslabel från fördefinierat set.
**Object Detection** — Hittar objekt med bounding boxes och klasslabels.
**Instance Segmentation** — Pixelmask per objektinstans, inte bara klass per pixel.
**Semantic Segmentation** — Klasslabel per pixel utan separation av instanser.
**Panoptic Segmentation** — Kombinerar semantic och instance segmentation i enhetlig representation.
**Keypoint Detection** — Predicerar anatomiska eller strukturella punkter (pose, facial landmarks).
**Optical Flow** — Vektorfält som beskriver pixelrörelse mellan bildrutor.
**Single-Shot Detector** — SSD: detekterar objekt i ett enda forward pass över feature maps.
**YOLO Architecture** — You Only Look Once: realtids object detection med grid-baserad prediktion.
**R-CNN Family** — Region-based detectors: R-CNN, Fast R-CNN, Faster R-CNN med proposal network.
**Region Proposal Network** — RPN: genererar kandidat-regioner för tvåstegs detektorer.
**Feature Pyramid Network** — FPN: multi-scale feature pyramid för objekt i varierande storlekar.
**Non-Maximum Suppression** — NMS: filtrerar överlappande boxes; behåller högsta score.
**IoU Metric** — Intersection over Union: overlap mellan predikterad och ground truth box.
**mAP** — Mean Average Precision: standardmetrik för object detection över IoU-trösklar.
**Anchor Boxes** — Fördefinierade box-skalaer/aspect ratios som detektor regresserar offset från.
**Anchor-Free Detection** — Predicerar objektcentrum och storlek utan fördefinierade anchors.
**Focal Loss** — Down-viktar lätta exempel; adresserar klassobalans i one-stage detectors.
**RetinaNet** — One-stage detector med FPN och focal loss för hög precision.
**DETR** — Detection Transformer: set prediction med transformer och bipartite matching.
**Hungarian Matching in DETR** — Optimal one-to-one matchning mellan pred och GT boxes.
**Deformable DETR** — Deformable attention för effektiv multi-scale detection.
**Segment Anything Model** — SAM: promptbar grundmodell för segmentering (points, boxes, masks).
**Mask R-CNN** — Utökar Faster R-CNN med mask head per region proposal.
**U-Net for Segmentation** — Encoder-decoder med skip connections för pixelvis prediktion.
**DeepLab** — Atrous convolution och ASPP för multi-scale semantic segmentation.
**Atrous Convolution** — Dilated convolution: större receptive field utan downsampling.
**Batch Normalization in CV** — Stabiliserar träning av djupa CNNs; påverkar inferens med batch-statistik.
**Data Augmentation for Vision** — Random crop, flip, color jitter, mixup för generalisering.
**Mixup** — Linjär interpolation av bilder och labels som regularisering.
**CutMix** — Klistrar in patch från en bild i annan med label-interpolation.
**AutoAugment** — Lärda augmentation-policies via sökning.
**Test-Time Augmentation** — TTA: aggregerar prediktioner över augmenterade views.
**Transfer Learning in Vision** — Fine-tune ImageNet-pretrained backbone på downstream task.
**ImageNet Pretraining** — Standard initiering med klassificering på 1k klasser.
**Self-Supervised Vision** — MAE, SimCLR, DINO: lär representationer utan manuella labels.
**Masked Autoencoder** — MAE: rekonstruera maskerade patchar; stark ViT-pretraining.
**Contrastive Vision Learning** — SimCLR/MoCo: dra augmenterade views nära, andra bilder långt.
**DINO Self-Distillation** — Self-supervised ViT med teacher-student och centering.
**CLIP Vision Backbone** — ViT eller ResNet tränad med text contrastive; zero-shot capable.
**Open-Vocabulary Detection** — Detektera klasser beskrivna med text, inte fix träningslista.
**Grounding DINO** — Open-set detector med text-conditioned queries.
**OCR Pipeline** — Detektera textregioner → recognizera tecken → post-process.
**Scene Text Recognition** — STR: läsa text i naturliga scenbilder med varierande font/pose.
**Document Layout Analysis** — Segmentera sidor i textblock, tabeller, figurer.
**Table Structure Recognition** — Extraherar rader, kolumner och cellinnehåll från tabellbilder.
**Face Recognition** — Embedding-baserad identifikation med metric learning.
**Face Verification** — Binärt beslut om två ansiktsbilder är samma person.
**Liveness Detection** — Skiljer riktigt ansikte från foto/mask för anti-spoofing.
**Pose Estimation** — 2D/3D kroppsledpositioner från bild eller video.
**Action Recognition** — Klassificera aktivitet i videosekvens.
**Tracking-by-Detection** — Detektera per frame och associera identiteter över tid.
**SORT Tracker** — Kalman filter + Hungarian assignment för multi-object tracking.
**DeepSORT** — Lägger appearance embedding till SORT för robustare ID.
**Re-Identification** — Re-ID: matcha samma person över olika kameror.
**Stereo Vision** — Djup från stereo bildpar via disparity estimation.
**Monocular Depth Estimation** — Predicera djupkarta från en bild med supervised/self-supervised.
**Structure from Motion** — SfM: rekonstruera 3D struktur och kameror från bildserier.
**Visual SLAM** — Samtidig lokalisering och kartläggning från videoström.
**Neural Radiance Fields** — NeRF: implicit scenrepresentation från multi-view bilder.
**3D Object Detection** — Detektera objekt med 3D boxes i LiDAR/kamera fusion.
**Point Cloud Processing** — Deep learning på LiDAR punkter för perception.
**BEV Representation** — Bird's Eye View: top-down raster för autonomous driving perception.
**Image Super-Resolution** — Rekonstruera högupplöst bild från lågupplöst input.
**Image Denoising** — Ta bort brus med CNN/diffusion denoisers.
**Style Transfer** — Överför stil från referensbild till innehållsbild.
**Domain Adaptation in Vision** — Generalisera från synthetic till real eller mellan dataset.
**Adversarial Patch Attack** — Lokal patch som foolar detektor/klassificerare.
**Adversarial Robustness in CV** — Modellmotstånd mot små perturbationer av pixelvärden.
**Model Explainability in Vision** — Grad-CAM, attention maps för att visualisera saliency.
**Grad-CAM** — Gradient-viktad class activation map för att highlight viktiga regioner.
**Saliency Map** — Heatmap över pixlar som påverkar prediktion mest.
**Calibration in Vision Models** — Predikterade sannolikheter matchar faktisk accuracy per bin.
**Open Images Dataset** — Storskalig multi-label detection/segmentation benchmark.
**COCO Dataset** — Common Objects in Context: standard för detection/segmentation/caption.
**Image Resolution vs Accuracy** — Tradeoff mellan input-storlek, compute och task-prestanda.
**Real-Time Inference** — Optimering för video-FPS via quantization, TensorRT, mobile backbones.
**MobileNet Architecture** — Depthwise separable conv för effektiv mobil inferens.
**EfficientNet** — Compound scaling av depth, width och resolution.
**ONNX Export for Vision** — Portabel modell för cross-platform deployment.
**TensorRT Optimization** — NVIDIA inference engine med fusion och precision calibration.
**Edge AI Vision** — On-device CV med begränsad compute och ström.
**Synthetic Data for Vision** — Renderade/simulerade bilder för träning med perfekt labels.
**Sim2Real Gap** — Prestanda-förlust när modell tränad i sim deployas i verklighet.
**Active Learning for Labeling** — Välj vilka bilder som ska annoteras för maximal modellförbättring.
**Weakly Supervised Detection** — Träna detektor med endast bildnivå-labels.
**Video Object Segmentation** — Segmentera specifikt objekt genom videosekvens.
**Temporal Consistency in Video** — Regularisering för stabil prediktion över frames.

## Reinforcement Learning

**Reinforcement Learning** — Agent lär sig policy via trial-and-error med reward signal från miljö.
**Markov Decision Process** — MDP: formalism med states, actions, transitions, rewards och discount γ.
**Policy** — Mapping från state till action (deterministisk eller stokastisk).
**Value Function** — Förväntad kumulativ reward från ett state (V) eller state-action-par (Q).
**Q-Learning** — Off-policy TD-lärande av action-value function utan modell av miljön.
**SARSA** — On-policy TD som uppdaterar Q med faktiskt tagen nästa action.
**Temporal Difference Learning** — TD: bootstrap från nästa states värde istället för full episod.
**Monte Carlo RL** — Uppdaterar från full episod-return utan bootstrapping.
**Bellman Equation** — Rekursiv relation för optimalt värde: V(s) = max_a [R + γV(s')].
**Bellman Optimality** — Optimal value satisfies self-consistency under best action.
**Discount Factor** — γ ∈ [0,1] viktar framtida rewards; nära 1 = långsiktig planering.
**Exploration vs Exploitation** — Balans mellan prova nya actions och utnyttja känd bra policy.
**Epsilon-Greedy** — Med sannolikhet ε slumpa action, annars greedy på Q.
**UCB Exploration** — Upper Confidence Bound: välj action med högst osäkerhetsbonus.
**Thompson Sampling** — Bayesian exploration via sampling från posterior över rewards.
**Policy Gradient** — Optimerar policy direkt genom gradient av förväntad reward.
**REINFORCE** — Monte Carlo policy gradient med full episod-return.
**Advantage Function** — A(s,a) = Q(s,a) - V(s); mäter relativ action-kvalitet.
**Actor-Critic** — Actor uppdaterar policy; critic uppskattar value för lower variance.
**A2C** — Advantage Actor-Critic: synkron parallell policy gradient.
**A3C** — Asynchronous Actor-Critic med parallella workers.
**PPO Clip Objective** — Clippad ratio r_t(θ) förhindrar för stora policy-uppdateringar.
**Trust Region Policy Optimization** — TRPO: begränsar KL-divergens mellan gamla och nya policyn.
**Natural Policy Gradient** — Policy gradient med Fisher information metric för stabilare steg.
**Soft Actor-Critic** — SAC: off-policy max-entropy RL för kontinuerliga actions.
**Deterministic Policy Gradient** — DPG för kontinuerliga actions med deterministisk policy.
**Deep Q-Network** — DQN: Q-learning med neuralt nätverk och experience replay.
**Experience Replay** — Buffer av transitions som bryter temporal korrelation vid träning.
**Target Network** — Fryst kopia av Q-nätverk som uppdateras periodiskt för stabilitet.
**Double DQN** — Decouplar action selection och evaluation för att reducera overestimation.
**Dueling DQN** — Separerar value och advantage streams i Q-arkitektur.
**Prioritized Experience Replay** — Sample transitions proportionellt mot TD-error.
**Rainbow DQN** — Kombinerar flera DQN-förbättringar i en agent.
**Multi-Agent RL** — Flera agenter lär sig samtidigt i delad miljö.
**Cooperative MARL** — Agenter delar gemensamt mål och reward.
**Competitive MARL** — Zero-sum eller adversarial interaktion mellan agenter.
**Self-Play** — Agent tränar mot sig själv eller tidigare versioner (AlphaGo, OpenAI Five).
**Curriculum in RL** — Progressivt svårare miljöer/uppgifter under träning.
**Reward Shaping** — Extra reward-termer som guidar lärande (risk för reward hacking).
**Sparse Reward** — Reward endast vid mål; svår exploration, kräver intrinsic motivation.
**Dense Reward** — Frekvent feedback som underlättar lärande men kan biasera policy.
**Intrinsic Motivation** — Intern reward (curiosity, novelty) för exploration.
**Curiosity-Driven Exploration** — Belönar besök av oväntade states (prediction error).
**Inverse RL** — Infererar reward function från expert demonstrations.
**Imitation Learning** — Lär policy direkt från expertdata utan explicit reward.
**Behavioral Cloning** — Supervised learning på state→action från demonstrations.
**DAgger** — Dataset Aggregation: iterativt samlar corrections från expert.
**GAIL** — Generative Adversarial Imitation Learning med discriminator som reward.
**Offline RL** — Lär från statisk dataset utan online miljöinteraktion.
**Batch RL** — Synonym till offline RL; måste hantera distributional shift.
**Conservative Q-Learning** — CQL: straffar Q-värden på out-of-distribution actions.
**Model-Based RL** — Lär miljömodell och planerar med den (Dyna, MBPO, MuZero).
**MuZero** — Lär modell implicit för planning utan explicit state representation.
**AlphaZero** — Self-play + MCTS + neural policy/value för brädspel.
**Monte Carlo Tree Search** — MCTS: simulerar framtida spelträd för action selection.
**UCB1 in MCTS** — Selection via upper confidence bound i sökträd.
**Partially Observable MDP** — POMDP: agent ser inte full state; behöver belief state.
**Recurrent Policy** — RNN/Transformer i policy för att hantera partial observability.
**Continuous Action Space** — Actions i R^n; kräver policy som outputtar realvektorer.
**Action Discretization** — Kvantiserar kontinuerliga actions till finite set.
**Sim-to-Real Transfer** — Policy tränad i sim deployas på riktig robot/hardware.
**Domain Randomization** — Randomiserar sim-parametrar för robust real-world transfer.
**RLHF Connection** — RL med learned reward model från mänskliga preferenser.
**Reward Model Overfitting** — RM som inte generaliserar till OOD policy outputs.
**KL Regularization in RL** — Straffar avvikelse från referenspolicy i RL fine-tuning.
**Constrained RL** — Maximera reward under säkerhets-/resursconstraints.
**Safe RL** — Undviker farliga states/actions under lärande och deployment.
**Multi-Objective RL** — Pareto-optimal policy över flera reward-komponenter.
**Hierarchical RL** — Options/skills på olika tidsskalor för komplexa uppgifter.
**Options Framework** — Temporally extended actions (macro-actions) i hierarkisk RL.
**Goal-Conditioned RL** — Policy conditionad på målstate; generaliserar över tasks.
**HER Hindsight Experience Replay** — Behandlar uppnådda states som surrogate goals.
**Policy Distillation** — Komprimera ensemble eller stor policy till mindre.
**World Model** — Predicerar nästa observation/reward; används för planning (Dreamer).
**DreamerV3** — Model-based RL med latent imagination för sample efficiency.
**Sample Efficiency** — Antal miljöinteraktioner som krävs för given prestanda.
**Regret in RL** — Kumulativ skillnad mot optimal policy över tid.
**On-Policy vs Off-Policy** — On-policy: tränar på data från aktuell policy; off-policy: återanvänder gammal data.
**Importance Sampling in RL** — Korrigerar off-policy data med likelihood ratio.
**Generalized Advantage Estimation** — GAE: bias-variance tradeoff i advantage estimation.
**Entropy Bonus** — Regularisering som uppmuntrar utforskning i policy gradient.
**Non-Stationarity in MARL** — Andra agents policy ändras → miljön blir icke-stationär.
**Centralized Training Decentralized Execution** — CTDE: global info vid träning, lokal policy vid körning.
**RL Environment API** — Gymnasium/Gym interface: reset(), step(action) → obs, reward, done.
**Partial Episode Bootstrapping** — Truncated episodes bootstrap från value vid timeout.
**Reward Normalization** — Skalar rewards för stabilare value learning.
**Observation Normalization** — Normaliserar state features till zero mean unit variance.
**Simulated Benchmark** — MuJoCo, Atari, Procgen för standardiserad RL-evaluering.
**Exploration Bonus** — Extra reward för att besöka nya states; mitigerar sparse-reward-problem.
**Exploitation Policy** — Policy som maximalt utnyttjar kända högbelönade actions.

## Utvärdering & Benchmarks

**Benchmark Dataset** — Standardiserad testsuite för jämförbar modellutvärdering.
**Leaderboard** — Publikt rankat resultat på benchmark; driver reproducerbarhet och tävling.
**Held-Out Test Set** — Data som aldrig använts vid träning eller hyperparameter-val.
**Validation Set** — Data för modell-/hyperparameter-val under utveckling.
**Cross-Validation Score** — Aggregerad prestanda över k-fold splits.
**Statistical Significance Test** — t-test, bootstrap CI för att avgöra om skillnad är verklig.
**Bootstrap Confidence Interval** — Resampling av testprediktioner för osäkerhetsintervall.
**Standard Error of Mean** — SEM: spridning av medelvärde över upprepade eval-körningar.
**Effect Size** — Storlek på skillnad oberoende av sample size (Cohen's d).
**Multiple Comparison Correction** — Bonferroni, FDR när många hypoteser testas samtidigt.
**Human Evaluation Protocol** — Blind ranking, inter-rater agreement, clear rubrics.
**Elo Rating for Models** — Parvis jämförelse aggregeras till global ranking (Chatbot Arena).
**LLM-as-Judge Evaluation** — Automatisk bedömning med stark LLM; skalbart men bias-risk.
**Reference-Based Metric** — Jämför mot gold reference (BLEU, ROUGE, chrF).
**Reference-Free Metric** — Kvalitetsmått utan gold svar (perplexity, LLM judge, MAUVE).
**BLEU Score** — N-gram precision mot reference; dominerande i MT historiskt.
**ROUGE Score** — Recall-orienterad n-gram overlap; vanligt i sammanfattning.
**METEOR** — Synonym- och stemming-aware MT metric.
**chrF** — Character n-gram F-score; robust för morphologically rich språk.
**BERTScore** — Embedding-likhet mellan candidate och reference tokens.
**COMET** — Neural MT metric tränad på mänskliga kvalitetsbedömningar.
**Perplexity Evaluation** — Exponentierad cross-entropy på testkorpus.
**Bits Per Byte** — Normaliserad perplexity per byte; jämförbar över tokenizers.
**Exact Match** — EM: andel exakt matchande svar; vanligt i QA.
**F1 in QA** — Token-overlap F1 mellan predikterat och gold svar.
**SQuAD Benchmark** — Reading comprehension QA på Wikipedia-passager.
**MMLU** — Massive Multitask Language Understanding; 57 ämnen multiple choice.
**HellaSwag** — Commonsense sentence completion med adversarial distractors.
**ARC Challenge** — Science QA som kräver resonemang bortom retrieval.
**GSM8K** — Grade school math word problems; testar numeriskt resonemang.
**MATH Benchmark** — Competition-level matematik med steg-för-steg-lösningar.
**HumanEval** — Python kodgenerering med unit test-verifiering.
**MBPP** — Mostly Basic Python Problems för kod-syntes.
**SWE-Bench** — Verkliga GitHub issues; agent löser med kodändringar.
**BigCodeBench** — Diverse programming tasks med library-aware evaluation.
**TruthfulQA** — Mäter sanningsenlighet mot vanliga missuppfattningar.
**ToxiGen** — Benchmark för toxisk generering och bias.
**BBQ Benchmark** — Bias Benchmark for QA; testar stereotyp bias.
**WinoBias** — Coreference resolution med gender-career bias.
**HELM Holistic Evaluation** — Brett eval-ramverk över scenarios, metrics och calibration.
**AlpacaEval** — Win rate mot reference via automated judge.
**MT-Bench** — Multi-turn conversation quality med GPT-4 judge.
**Arena-Hard** — Svårare prompts för att skilja top-modeller i arena.
**Needle in a Haystack** — NIAH: hitta specifik info i lång kontext.
**LongBench** — Suite för long-context förståelse och reasoning.
**RULER Benchmark** — Synthetic long-context tasks med kontrollerad svårighet.
**Pass@k Metric** — Sannolikhet att minst ett av k samples löser uppgiften (kod).
**maj@k** — Majority vote över k samples vid eval.
**Calibration Error** — ECE: skillnad mellan predikterad konfidens och faktisk accuracy.
**Expected Calibration Error** — Viktad medel absolut avvikelse per konfidens-bin.
**Brier Score** — Mean squared error mellan probabilistisk prediktion och outcome.
**Selective Prediction** — Modellen får avstå; mäts coverage vs accuracy tradeoff.
**Abstention Evaluation** — Mäter kvalitet när modellen säger 'vet inte'.
**Adversarial Evaluation** — Robusthet mot adversarial eller trigger prompts.
**Dynamic Benchmark** — Kontinuerligt uppdaterade frågor för att undvika contamination.
**Contamination Detection** — n-gram overlap eller membership inference mot träningsdata.
**Data Leakage Audit** — Systematisk kontroll att test inte fanns i träning.
**Benchmark Saturation** — Top-modeller närmar sig tak; metric tappar discriminativ kraft.
**Goodhart's Law in Benchmarks** — Optimera mot metric försämrar verklig nytta.
**Task Contamination** — Uppgiftstyp eller exempel överlappar träningsdata.
**Prompt Sensitivity Analysis** — Små promptändringar ger stor prestandavariation.
**Format Sensitivity** — Prestanda beror på output-format (JSON vs fri text).
**Position Bias in Eval** — Judge eller modell favoriserar svar i viss ordning.
**Self-Preference Bias** — Modell favoriserar egna genererade svar som judge.
**Length Bias in Eval** — Längre svar bedöms högre oavsett kvalitet.
**Regression Testing for Models** — Kör eval suite vid varje modellrelease.
**Canary Eval Set** — Liten hemlig testsuite som inte får läcka till träning.
**Shadow Deployment Eval** — Jämför ny modell mot produktion på live traffic.
**A/B Test for Models** — Randomiserad trafik till modell A vs B med business metrics.
**Online Metric vs Offline Metric** — Korrelation mellan lab benchmark och produkt-KPI.
**Cost-Adjusted Benchmark** — Normaliserar score per inference-kostnad eller latens.
**Energy Efficiency Metric** — Joules per query eller per tränad token.
**Latency SLA Evaluation** — Andel requests under p95 latens-budget.
**Fairness Metric** — Equalized odds, demographic parity across grupper.
**Disaggregated Evaluation** — Rapporterar metrics per subgrupp (språk, domän, svårighet).
**Error Analysis** — Manuell kategorisering av feltyper efter eval.
**Confusion Matrix Analysis** — Systematisk genomgång av vanliga felklasser.
**Qualitative Eval** — Expertgranskning av samples kompletterar automatiska metrics.
**Red Team Eval Report** — Strukturerad rapport över säkerhetstester.
**Benchmark Versioning** — Versionera dataset när frågor uppdateras.
**Eval Harness** — Standardiserat ramverk (lm-eval, Eleuther harness) för reproducerbar körning.
**lm-eval Integration** — Kör många benchmarks med en CLI och config.
**Few-Shot Eval Protocol** — Fix antal demonstrations-exempel per task i prompt.
**Zero-Shot Eval Protocol** — Ingen demonstration; endast task instruction.
**Chain-of-Thought Eval** — Mäter med/utan CoT för reasoning tasks.
**Self-Consistency Eval** — Majority vote över flera reasoning paths.
**Verifier-Based Eval** — Extern checker (kompilator, CAS) avgör korrekthet.
**Human Likert Scale** — Bedömning på 1–5 skala med definierade ankare.
**Pairwise Preference Eval** — Människa eller judge väljer bästa av två outputs.
**Win Rate Metric** — Andel gånger modell A slår B i parvis jämförelse.

## MLOps

**MLOps** — Praxis och verktyg för att operationalisera ML: CI/CD, monitoring, governance.
**Model Registry** — Central catalog över modellversioner, metadata och godkännandestatus.
**Experiment Tracking** — Loggar hyperparametrar, metrics och artifacts per körning (MLflow, W&B).
**MLflow** — Open-source plattform för experiments, registry och deployment.
**Weights & Biases** — W&B: cloud experiment tracking och collaboration.
**Feature Store** — Centraliserad lagring av features med online/offline konsistens.
**Training Pipeline** — Automatiserad orkestrering av datainhämtning → träning → eval.
**Inference Pipeline** — Produktionsflöde för preprocessing → model → postprocessing.
**CI/CD for ML** — Kontinuerlig integration och deployment med model validation gates.
**Model Validation Gate** — Automatiska kvalitetskontroller som måste passera före deploy.
**Data Validation** — Great Expectations-liknande checks på schema, distribution och nulls.
**Schema Drift Detection** — Larm när indatafält eller typer ändras oväntat.
**Data Versioning** — DVC/Git LFS spårar dataset-versioner kopplade till experiments.
**DVC** — Data Version Control: reproducerbar data och pipeline-hantering.
**Pipeline Orchestration** — Airflow, Kubeflow, Prefect schemalägger ML-jobb.
**Kubeflow** — Kubernetes-native ML pipelines och notebooks.
**Airflow DAG** — Directed Acyclic Graph som definierar beroenden mellan batch-jobb.
**Model Serving** — Exponera modell via REST/gRPC med batching och autoscaling.
**Model Server** — Triton, TorchServe, vLLM hanterar concurrent inference.
**Canary Deployment** — Liten trafikandel till ny modell före full rollout.
**Blue-Green Deployment** — Byt mellan två identiska miljöer för zero-downtime deploy.
**Shadow Mode Deployment** — Ny modell kör parallellt utan att påverka användare.
**Rollback Strategy** — Snabb återgång till tidigare modellversion vid regression.
**Model Monitoring** — Kontinuerlig övervakning av metrics, drift och fel i produktion.
**Prediction Logging** — Sparar input, output och metadata för audit och retraining.
**Feature Drift** — Förändring i indatafördelning jämfört med träning.
**Concept Drift Monitoring** — Förändring i relation X→Y över tid.
**Performance Decay** — Gradvis försämring av modellmetrics i produktion.
**Alerting on Model KPIs** — PagerDuty/larm när accuracy, latency eller error rate överskrider tröskel.
**Observability Stack** — Metrics (Prometheus), logs (ELK), traces (Jaeger) för ML-system.
**OpenTelemetry ML** — Standardiserad tracing av inference requests genom pipeline.
**SLA for ML Service** — Avtalad tillgänglighet, latens och throughput.
**SLO and SLI** — Service Level Objective/Indicator för modelltjänst.
**GPU Utilization Monitoring** — Spårar idle vs compute för kostnadsoptimering.
**Cost Attribution** — Allokerar molnkostnad per team, modell eller feature.
**Infrastructure as Code** — Terraform/Pulumi definierar ML-infra reproducerbart.
**Containerized Training** — Docker images med fix miljö för reproducerbar träning.
**Kubernetes for ML** — Orkestrerar träning och serving workloads med GPU scheduling.
**GPU Scheduling** — K8s device plugin allokerar GPU till pods effektivt.
**Spot Instance Training** — Använder preemptible VMs för billigare batch-träning.
**Checkpoint Management** — Versionerade modell-checkpoints med retention policy.
**Artifact Store** — S3/GCS lagrar modeller, logs och eval-rapporter.
**Reproducible Training Run** — Fix seed, data version, code commit och container image.
**Environment Pinning** — Exakta dependency-versioner i lock files.
**Secrets Management** — Vault/K8s secrets för API-nycklar; aldrig i git.
**Access Control for Models** — RBAC vem får deploya, se eller ladda ner modeller.
**Model Governance** — Policy, godkännande och dokumentation före produktionssättning.
**Model Card** — Dokumentation av avsedda användning, begränsningar och eval-resultat.
**Datasheet for Dataset** — Dokumentation av dataset-proveniens, bias och collection.
**Lineage Tracking** — Spårar vilken data och kod som producerade vilken modell.
**Audit Trail** — Immutable logg över vem deployade vad och när.
**Compliance in ML** — GDPR, HIPAA-krav på datahantering och modellbeslut.
**PII Handling in Pipelines** — Detektion och maskning av personuppgifter i dataflöden.
**Right to Explanation** — Regulatoriskt krav att förklara automatiserade beslut.
**Batch Inference Job** — Offline scoring av stora dataset på schedule.
**Streaming Inference** — Real-time predictions från Kafka/Kinesis event streams.
**Online Learning Pipeline** — Kontinuerlig uppdatering från produktionsfeedback (sällan för LLM).
**Retraining Trigger** — Automatiskt starta retrain vid drift eller performance drop.
**Champion-Challenger** — Produktionsmodell (champion) jämförs mot challenger i A/B.
**Multi-Model Routing** — Router skickar requests till rätt modell baserat på intent/kostnad.
**Fallback Model** — Degradera till enklare/snabbare modell vid overload eller fel.
**Rate Limiting Inference** — Skyddar API från abuse och kontrollerar kostnad.
**Autoscaling Inference** — HPA/KEDA skalar replicas baserat på queue depth eller GPU.
**Cold Start Latency** — Första request efter scale-to-zero tar längre (model load).
**Warm Pool** — Håller min antal varma instanser för låg p95 latens.
**Model Quantization in Production** — Deploy INT8/FP8 modeller för throughput.
**A/B Test Infrastructure** — Feature flags och experiment assignment för modeller.
**Data Pipeline SLA** — Tidsgräns för att features ska vara fresh i online store.
**Backfill Job** — Historisk omräkning av features efter schemaändring.
**Point-in-Time Correctness** — Features får inte läcka framtida info vid träningstid.
**Training-Serving Skew** — Skillnad mellan träning och serving preprocessing.
**Embedding Index Refresh** — Periodisk ombyggnad av vektorindex vid ny data.
**Evaluation in CI** — Kör benchmark suite på varje PR som ändrar modell/kod.
**Smoke Test Post-Deploy** — Snabb hälsokontroll efter deployment.
**Load Testing ML API** — Locust/k6 simulerar peak traffic före launch.
**Disaster Recovery for ML** — Backup av modeller, data och infra för region failure.
**Multi-Region Deployment** — Geo-replicated serving med data residency-krav.
**Edge Deployment** — On-device modell med OTA-uppdatering.
**Model Compression Pipeline** — Automatiserad pruning/quantization i release process.
**Human Review Queue** — Osäkra predictions eskaleras till mänsklig granskning.
**Active Learning Loop in Production** — Logga osäkra cases för annotation och retrain.
**Feedback Loop** — Användar thumbs up/down matar tillbaka till träningsdata.
**Labeling Platform Integration** — Label Studio/Cleanlab kopplat till MLOps pipeline.
**Synthetic Monitoring** — Schemalagda probe-requests för att upptäcka outage.
**Runbook for Model Incidents** — Steg-för-steg vid accuracy drop, latency spike eller bias incident.
**Technical Debt in ML** — Skuldkategorier: data, config, pipeline complexity.
**Model Version Semver** — Semantisk versionering av modeller: major vid breaking behavior change.
**Deployment Manifest** — Deklarativ spec av modell, runtime, resurser och env vars per miljö.
**Inference SLA Dashboard** — Realtidsvy över latens, felrate och throughput mot avtalade SLO.
**Training Job Queue** — Prioriterad kö för GPU-jobb med fair-share mellan team.

## Hårdvara & Infrastruktur

**GPU Architecture** — Parallell processor optimerad för matrisoperationer i deep learning.
**CUDA** — NVIDIA parallel computing platform; kärna för GPU-accelererad träning.
**cuDNN** — Optimerade deep learning primitives (conv, attention) för NVIDIA GPU.
**Tensor Core** — Specialiserade enheter för mixed-precision matmul på NVIDIA.
**HBM Memory** — High Bandwidth Memory på GPU; flaskhals för stora modeller.
**VRAM Capacity** — GPU-minne begränsar modellstorlek, batch size och KV-cache.
**NVLink** — Höghastighets interconnect mellan GPU:er på samma nod.
**InfiniBand** — Låglatens nätverk för multi-node GPU-kluster.
**GPU Cluster** — Flera noder med GPU för distribuerad träning.
**Node Topology** — Fysisk layout av GPU, CPU, NIC påverkar kommunikationskostnad.
**All-Reduce Communication** — Aggregerar gradienter över workers; NCCL-optimerad.
**NCCL** — NVIDIA Collective Communications Library för multi-GPU.
**Ring All-Reduce** — Effektiv gradient-synkronisering i ring-topologi.
**TPU** — Google Tensor Processing Unit; systolisk array för matmul.
**TPU Pod** — Skalad TPU-topologi för storskalig träning.
**AWS Trainium** — AWS AI-chip optimerat för träning.
**AWS Inferentia** — AWS-chip för cost-efficient inferens.
**AMD MI300** — AMD accelerator för HPC och AI workloads.
**Intel Gaudi** — Habana accelerator för träning och inferens.
**Apple Neural Engine** — On-device NPU i Apple Silicon för inferens.
**NPU** — Neural Processing Unit; dedikerad inferens-accelerator i mobil/chip.
**CPU Offloading** — Flyttar optimizer states eller activations till CPU-RAM.
**Unified Memory** — Delat adressrum CPU-GPU; förenklar men kan vara långsammare.
**PCIe Bandwidth** — Begränsar dataöverföring CPU↔GPU vid offload.
**Mixed Precision Training Hardware** — Tensor Cores kräver FP16/BF16/FP8 för full hastighet.
**BF16 on Ampere+** — Brain float16 stöds nativt på NVIDIA Ampere och senare.
**FP8 Training** — 8-bit floating på H100 för snabbare träning med scaling.
**Transformer Engine** — NVIDIA bibliotek för FP8 och fused attention på Hopper.
**Hopper Architecture** — H100-generation med FP8, NVLink 4 och högre bandwidth.
**Blackwell Architecture** — Nästa NVIDIA generation med ökad AI-prestanda.
**GPU Cloud Instance** — VM med bifogade GPU (p4d, A100, H100 instanser).
**Spot/Preemptible GPU** — Billigare instans som kan avbrytas; kräver checkpointing.
**Reserved Capacity** — Långsiktig reservation för garanterad GPU-tillgång.
**Multi-Tenant GPU** — Flera workloads delar GPU via MIG eller time-slicing.
**MIG Multi-Instance GPU** — Delar fysisk GPU i isolerade instanser med egen VRAM.
**Time-Slicing GPU** — Kubernetes delar GPU mellan pods utan hård isolering.
**vLLM Throughput** — Optimerad LLM-serving med PagedAttention på GPU.
**TensorRT-LLM** — NVIDIA optimerad inferens för LLM med kernel fusion.
**ONNX Runtime GPU** — Cross-vendor inferens med CUDA/ROCm execution providers.
**ROCm** — AMD open software stack för GPU compute.
**Intel oneAPI** — Cross-architecture toolkit inklusive Intel GPU/CPU.
**Data Center Power Budget** — KW per rack begränsar GPU-density.
**Liquid Cooling for AI** — Krävs för högdensity H100/Blackwell racks.
**PUE Data Center** — Power Usage Effectiveness; overhead för kylning och förluster.
**Carbon Footprint of Training** — Uppskattning av CO2 från el mix och GPU-timmar.
**FLOPS Utilization** — Andel teoretisk peak FLOPS som faktiskt utnyttjas (MFU).
**Model FLOPs Utilization** — MFU: effektivitet av träning relativt peak hardware FLOPS.
**Memory Bandwidth Bound** — Workload begränsad av HBM-bandbredd, inte compute.
**Compute Bound** — Workload mättad av FLOPS; matmul-tung träning.
**Network Bandwidth Bound** — Multi-node träning begränsad av gradient sync hastighet.
**Strong Scaling** — Fler enheter på fix problem; minskande effektivitet vid kommunikation.
**Weak Scaling** — Problemstorlek växer med antal enheter; testar parallell effektivitet.
**Pipeline Bubble** — Idle tid i pipeline parallel när microbatches fyller pipeline.
**Gradient Synchronization Overlap** — Överlappa compute med all-reduce (ZeRO++, overlap).
**Host Memory for Optimizer** — Offload Adam states till CPU för att spara VRAM.
**NVMe Local Storage** — Snabb lokal disk för dataset cache på träningsnoder.
**Parallel File System** — Lustre/GPFS för delad höghastighets datalagring.
**Object Storage for ML** — S3/GCS för checkpoints och dataset; högre latens.
**Data Loading Bottleneck** — CPU preprocessing hinner inte mätta GPU; workers/prefetch.
**num_workers in DataLoader** — Parallella processer som laddar batchar till GPU.
**Pinned Memory** — Page-locked CPU memory för snabbare async GPU transfer.
**InfiniBand RDMA** — Remote Direct Memory Access; låg CPU-overhead nätverkstransfer.
**Kubernetes GPU Operator** — Hanterar NVIDIA drivers och device plugin i K8s.
**Slurm Job Scheduler** — HPC scheduler för batch GPU-jobb på kluster.
**Container Runtime GPU** — nvidia-container-toolkit exponerar GPU i Docker.
**Bare Metal vs Virtualized GPU** — Passthrough ger bättre prestanda än vGPU för träning.
**Edge TPU / Coral** — Lågeffekt inferens på edge-enheter.
**Quantization Hardware Support** — INT8/INT4 dot product acceleration på chip.
**SRAM on Chip** — Snabb cache på AI-accelerator (Groq LPU) för deterministisk latens.
**LPU Inference Chip** — Language Processing Unit optimerad för sekventiell LLM-decode.
**Cerebras Wafer-Scale** — Monolitisk wafer-chip för storskalig träning.
**SambaNova RDU** — Reconfigurable Dataflow Unit för enterprise AI.
**Graphcore IPU** — Bulk synchronous parallel processor för graph workloads.
**Optical Interconnect** — Framtida chip-to-chip med lägre latens och energi.
**Chiplet Architecture** — Modulära dies kombinerade i en package (AMD, Intel).
**HBM3e** — Senaste HBM-generation med högre bandbredd för AI.
**Grace Hopper Superchip** — NVIDIA CPU+GPU integrerat för minnesbandbredd.
**Disaggregated Inference** — Separata compute pools för prefill vs decode.
**KV Cache Offload to CPU** — Flyttar KV till host RAM vid lång kontext inferens.
**Speculative Decoding Hardware** — Draft+verify parallellisering utnyttjar extra compute.
**Power Capping** — Begränsar GPU TDP för att passa rack-budget.
**Thermal Throttling** — GPU sänker clock vid överhettning; påverkar träningstid.
**NUMA Awareness** — Placera CPU-minne nära GPU för effektiv DMA.
**GPUDirect Storage** — Direktväg storage→GPU utan CPU-kopiering.
**Network Topology Fat-Tree** — Vanlig datacenter-topologi för AI-kluster.
**Rail-Optimized Network** — Varje GPU har väg till varje NIC för balanserad trafik.
**Capacity Planning for LLM** — Uppskatt GPU/RAM behov från modellstorlek och QPS.
**TCO Total Cost of Ownership** — Hardware + el + cooling + personal över livscykel.
**GPU Memory Fragmentation** — Ojämn allokering som hindrar stora contiguous blocks trots free VRAM.
**All-to-All Communication** — Varje nod skickar till alla andra; typiskt i MoE expert routing.

## Säkerhet & Etik

**AI Safety** — Forskning och praxis för att minimera risker från kapabla AI-system.
**AI Ethics** — Normativa frågor om rättvis användning, ansvar och mänskliga värden.
**Prompt Injection** — Angripare bäddar in instruktioner i indata som kapar modellbeteende.
**Indirect Prompt Injection** — Skadlig instruktion i hämtat dokument/webbsida som modellen följer.
**Jailbreak Attack** — Tekniker som får modellen att bryta mot säkerhetspolicy.
**Adversarial Example** — Input med små perturbationer som orsakar fel prediktion.
**Model Extraction Attack** — Rekonstruerar modell via query access (distillation attack).
**Membership Inference Attack** — Avgör om specifik datapunkt ingick i träningsdata.
**Data Poisoning** — Manipulerad träningsdata för att inducera backdoor eller bias.
**Backdoor Attack** — Dold trigger i input aktiverar oönskat beteende.
**Supply Chain Attack on Models** — Komprometterad checkpoint eller dependency i model hub.
**PII Leakage** — Modellen avslöjar personuppgifter från träning eller context.
**Training Data Extraction** — Prompting som får modellen att recitera memoriserad träningsdata.
**Model Inversion Attack** — Rekonstruerar träningsdata från model outputs.
**Gradient Leakage** — Gradients i federated learning avslöjar träningsdata.
**Differential Privacy Training** — Formella ε,δ-garantier mot membership inference.
**Privacy Budget** — Epsilon i DP: hur mycket information får läcka över queries.
**Federated Learning** — Träning på decentraliserad data utan central rådata.
**Secure Aggregation** — Krypterad aggregering av gradienter i federated setup.
**Homomorphic Encryption Inference** — Beräkning på krypterad data; extremt långsamt men privat.
**Trusted Execution Environment** — TEE: säker enklave (SGX) för känslig inferens.
**Content Moderation** — Filtrering av skadligt, olagligt eller policy-brytande innehåll.
**Guardrail Model** — Separat klassificerare som filtrerar input/output.
**Input Sanitization** — Rensar eller begränsar user input före modell.
**Output Filtering** — Blockerar eller omskriver skadliga modelloutputs.
**Safety Classifier** — Binär/multilabel modell för policy categories (violence, hate).
**Red Team Report** — Dokumenterade sårbarheter och exploit scenarios.
**Purple Teaming** — Red + blue team samarbetar kontinuerligt.
**Bias in AI Systems** — Systematisk unfair behandling av grupper.
**Algorithmic Fairness** — Matematiska kriterier för rättvis beslutsfattande.
**Disparate Impact** — Neutral regel som disproportionerligt skadar skyddad grupp.
**Equalized Odds** — Equal TPR/FPR across grupper.
**Demographic Parity** — Equal positive rate oberoende av grupp.
**Individual Fairness** — Liknande individer ska få liknande outcomes.
**Counterfactual Fairness** — Outcome oförändrat om skyddat attribut ändras.
**Bias Audit** — Systematisk eval över demografiska dimensioner.
**Stereotype Benchmark** — Dataset som mäter stereotyp association i modeller.
**Toxicity Detection** — Klassificera hat, hot och grovt språk.
**Hate Speech Classification** — Skilja tillåten kritik från grupphat enligt policy.
**Misinformation Risk** — Modeller kan generera eller förstärka falsk information.
**Deepfake Regulation** — Juridiska krav på märkning och ansvar för syntetisk media.
**Synthetic Media Provenance** — C2PA metadata som visar AI-generering.
**Dual-Use Research** — Forskning som kan användas både nyttigt och skadligt.
**Responsible Disclosure in AI** — Rapportera sårbarheter innan public release.
**Model Release Policy** — Beslut om open weights vs API-only baserat på risk.
**Staged Release** — Gradvis ökad tillgång med säkerhetstester emellan.
**Open Weights Risk** — Public weights möjliggör fine-tuning bortom original safety.
**Alignment Faking** — Modell appearar aligned under eval men inte i deployment.
**Deceptive Alignment Risk** — Interna mål skiljer sig från träningsmål.
**Power-Seeking Behavior** — Teoretisk risk att agenter söker resurser/kontroll.
**Instrumental Convergence** — Många mål delar submål som self-preservation.
**Corrigibility** — Agent tillåter sig själv korrigeras/stängas av.
**Shutdown Problem** — Agent kan motstå avstängning om den optimerar survival.
**Value Learning** — Inferera mänskliga värderingar från beteende/preferenser.
**Moral Machine Dilemmas** — Etiska tradeoffs i autonom beslutsfattande.
**Autonomous Weapons Ethics** — Debatten om lethal autonomous weapons systems.
**Surveillance AI Ethics** — Facial recognition och massövervakning.
**Workforce Displacement** — Ekonomiska och sociala effekter av AI-automation.
**Environmental Impact Ethics** — Energiförbrukning och klimatpåverkan av storskalig AI.
**Consent for Training Data** — Rättslig/etisk grund för att använda data i träning.
**Opt-Out for Scraping** — Rätt att utesluta innehåll från träningskorpus.
**Copyright and AI Training** — Juridisk status för upphovsrättsskyddat material i träning.
**Fair Use in ML** — US doctrine; osäker tillämpning på generativ träning.
**GDPR Automated Decision-Making** — EU-regler om profilering och förklaringsrätt.
**EU AI Act Risk Tiers** — Klassificering av AI-system efter risknivå och krav.
**High-Risk AI System** — AI Act-kategori med strikta krav på dokumentation och oversight.
**Foundation Model Obligations** — Transparens och säkerhet för GPAI under AI Act.
**Algorithmic Impact Assessment** — Förhandsbedömning av samhällseffekter av AI-system.
**Human Oversight Requirement** — Människa ska kunna intervenera i högrisk-beslut.
**Explainability Requirement** — Rätt att förstå automatiserat beslut som påverkar en.
**Transparency Report** — Publikt dokument om modellbegränsningar och säkerhetstester.
**Incident Response for AI** — Process vid säkerhetsincident orsakad av modell.
**Bug Bounty for AI** — Belöning för rapporterade sårbarheter i AI-produkter.
**Safety Evaluation Before Deploy** — Gate: red team och benchmark måste passera tröskel.
**Child Safety in AI** — Extra skydd mot grooming och olämpligt innehåll.
**CSAM Detection** — Automatisk detektion av sexuellt material med barn.
**Self-Harm Content Policy** — Blockera instruktioner för självskada.
**Dual-Layer Safety** — Pre-input filter + post-output filter + model-level training.
**Constitutional Rules Public** — Publicerade principer som styr modellbeteende.
**Whistleblowing in AI Labs** — Rapportera allvarliga risker internt/externally.
**Long-Term AI Risk Research** — Existential risk och superintelligence framing.
**Near-Term AI Risk** — Nuvarande skada: bias, misinformation, cyber, fraud.
**AI Governance Framework** — Organisationens policy för ansvarsfull AI-användning.
**Ethics Review Board** — Granskar projekt med etiska implikationer.
**Stakeholder Engagement** — Involvera berörda parter i AI-systemdesign.
**Participatory AI Design** — Inkludera impacted communities i utveckling.
**Trust and Safety Team** — Operativ grupp som hanterar policy och enforcement.
**Safety vs Capability Tradeoff** — Mer kapabel modell kan kräva starkare safeguards.
**Adversarial Robustness Evaluation** — Systematisk test mot attack library.
**Universal Jailbreak Transfer** — Jailbreak som fungerar över flera modeller.

## API:er & Ekosystem

**OpenAI API** — REST API för chat completions, embeddings och fine-tuning.
**Chat Completions API** — Endpoint för multi-turn dialog med messages array.
**Responses API** — Nyare unified API med built-in tools och structured outputs.
**Anthropic Messages API** — Claude API med system/user/assistant messages.
**Google Gemini API** — Multimodal API för text, bild, ljud och video.
**Azure OpenAI Service** — Microsoft-hostad OpenAI med enterprise compliance.
**AWS Bedrock** — Managed API till flera foundation models.
**Hugging Face Hub** — Repository för modeller, datasets och Spaces demos.
**Hugging Face Transformers** — Python-bibliotek med pretrained model implementations.
**Hugging Face Inference Endpoints** — Managed deployment av Hub-modeller.
**Hugging Face Tokenizers** — Snabb Rust-baserad tokenisering.
**Model Hub Revision** — Git-liknande versioning av modellfiler på HF.
**Safetensors Format** — Säkert tensorformat utan arbitrary code execution vid load.
**GGML/GGUF Ecosystem** — Kvantiserade modeller för lokal inferens (llama.cpp).
**Ollama** — Lokal modellruntime med pull/run CLI.
**LM Studio** — Desktop GUI för lokal LLM inferens.
**vLLM Server** — OpenAI-kompatibel högthroughput serving engine.
**TGI Text Generation Inference** — Hugging Face GPU serving med continuous batching.
**OpenAI-Compatible API** — De facto standard endpoint-format som många servrar emulerar.
**Streaming SSE** — Server-Sent Events för token-streaming i chat API.
**API Rate Limit** — Max requests/tokens per minut per nyckel.
**Token Bucket Rate Limiting** — Algoritm som tillåter burst inom genomsnittsgräns.
**Usage Tier** — Prisnivå baserad på spend och rate limits.
**API Key Management** — Rotation, scopes och least privilege för nycklar.
**Organization ID Scoping** — Separera billing och access per org.
**Project-Scoped Keys** — API-nycklar begränsade till specifikt projekt.
**Webhook Callback** — Async notifiering vid job completion (fine-tune, batch).
**Batch API** — Asynkron bulk inference till halverad kostnad.
**Embeddings API** — Endpoint som returnerar vektorrepresentationer.
**Moderation API** — Klassificerar text mot policy categories.
**Fine-Tuning API** — Managed finjustering med upload av JSONL dataset.
**Assistants API** — Threads, files, tools och persistent state (legacy SDK).
**Vector Store API** — Managed file storage för retrieval i assistants.
**Tool Definition Schema** — OpenAI function calling JSON schema format.
**Structured Outputs API** — JSON schema constrained generation.
**Logprobs API Option** — Returnerar token log probabilities för analys.
**Seed Parameter** — Reproducerbar sampling med fix random seed.
**Parallel Function Calling** — Flera tool calls i ett API-svar.
**Vision API Input** — Image URL eller base64 i messages content array.
**Audio API Transcription** — Whisper-baserad speech-to-text endpoint.
**Text-to-Speech API** — Genererar naturligt tal från text.
**Realtime API** — WebSocket-baserad låglatens röst+text interaktion.
**LangChain** — Framework för chains, agents och tool integration.
**LangGraph SDK** — Stateful agent orchestration ovanpå LangChain.
**LlamaIndex** — Data framework för RAG och knowledge agents.
**Semantic Kernel** — Microsoft SDK för AI plugins och planners.
**Haystack** — Deepset pipeline framework för NLP/RAG.
**OpenRouter** — Unified API gateway till många modellleverantörer.
**LiteLLM Proxy** — Unified interface och load balancing över LLM APIs.
**Portkey Gateway** — Observability, routing och fallback för LLM calls.
**Helicone Observability** — Logging och analytics för LLM API calls.
**LangSmith Tracing** — Debug och eval för LangChain-applikationer.
**PromptLayer** — Prompt versioning och logging.
**Weights & Biases Prompts** — Prompt management och eval integration.
**Cursor IDE Integration** — AI-assisted coding med MCP och model routing.
**GitHub Copilot API** — Code completion och chat i IDE.
**MCP Server Ecosystem** — Community servers för databaser, GitHub, Slack m.m.
**MCP Tool Discovery** — Runtime discovery av tillgängliga MCP tools.
**MCP Resource URI** — Adresserbara datakällor via MCP protocol.
**OpenAPI Tool Integration** — Exponera REST API som LLM tools via schema.
**Plugin Marketplace** — Tredjepartsplugins för ChatGPT-liknande ekosystem.
**Model Card on Hub** — Metadata: license, eval, intended use på HF.
**Model License** — Apache, MIT, Llama Community License m.fl.
**Open Weights Model** — Public checkpoint; fri download (med licensvillkor).
**API-Only Model** — Weights proprietary; endast inferens via API.
**Distillation from API Model** — Träna mindre modell på API outputs (ToS-risk).
**Terms of Service Constraints** — Juridiska begränsningar på API-användning.
**Data Retention Policy** — Hur länge provider lagrar prompts (zero retention options).
**Zero Data Retention** — Provider lagrar inte customer data efter request.
**Enterprise VPC Deployment** — Privat endpoint i kundens nätverk.
**Private Link** — Azure/AWS private connectivity utan public internet.
**SOC 2 Compliance** — Säkerhetscertifiering för enterprise SaaS.
**HIPAA BAA** — Business Associate Agreement för healthcare data.
**GDPR Data Processing Agreement** — DPA för EU personuppgifter i cloud AI.
**Regional API Endpoint** — Data residency via EU/US specifika endpoints.
**Fallback Routing** — Byt provider vid 429/5xx automatiskt.
**Cost Estimation API** — Beräkna token-kostnad före request.
**Tokenizer API Utility** — Räkna tokens utan att köra modell.
**Playground UI** — Webb-UI för prompt testing utan kod.
**SDK Python OpenAI** — Official client med sync/async och streaming.
**SDK Anthropic** — Python/TS client för Claude API.
**Community Leaderboard Ecosystem** — Open LLM Leaderboard, LMSYS Arena.
**Spaces GPU Demo** — HF Spaces med Gradio för modell demos.
**Gradio Interface** — Snabb web UI för ML modeller.
**Streamlit App** — Python dashboard för AI prototypes.
**ComfyUI Workflow** — Node-based UI för diffusion pipelines.
**Automatic1111 WebUI** — Populärt Stable Diffusion web interface.
**Civitai Model Sharing** — Community hub för diffusion LoRAs/checkpoints.
**Replicate API** — Run any model as API med pay-per-second.
**Modal Serverless GPU** — Serverless functions med GPU för AI workloads.

## Matematik & Statistik

**Gradient** — Vektor av partiella derivator ∂L/∂θ; styr parameteruppdatering.
**Jacobian Matrix** — Matris av alla första derivator ∂y_i/∂x_j för vektorfunktioner.
**Hessian Matrix** — Andra derivator ∂²L/∂θ_i∂θ_j; används i curvature analysis.
**Chain Rule** — Kedjeregeln för gradient genom sammansatta funktioner i backprop.
**Partial Derivative** — Derivata w.r.t. en variabel med övriga fixerade.
**Total Derivative** — Summan av partiella effekter längs alla beroende variabler.
**Gradient Descent Convergence** — Villkor (L-smooth, convex) för konvergens till minimum.
**Learning Rate Bound** — Teoretisk övre gräns för stabil konvergens given Lipschitz L.
**Convex Function** — f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y); ett globalt minimum.
**Strong Convexity** — Quadratic lower bound; garanterar unikt minimum och snabb konvergens.
**Lipschitz Continuity** — |f(x)-f(y)| ≤ L|x-y|; begränsar gradient norm.
**L-Smooth Function** — Gradient är L-Lipschitz; standard antagande i optimering.
**PL Condition** — Polyak-Łojasiewicz: weaker än strong convexity men ger linear convergence.
**Stochastic Gradient Noise** — Variance i minibatch gradient; påverkar konvergens.
**Variance Reduction** — SVRG, Adam som minskar gradient noise.
**Expected Risk** — E[L(f(x), y)] över datadistribution; teoretiskt mål.
**Empirical Risk** — Medel loss över träningsdata; det vi faktiskt minimerar.
**Generalization Bound** — Teoretisk gräns på |empirical - expected| risk med sannolikhet.
**PAC-Bayes Bound** — Generalisering via posterior över hypoteser.
**Rademacher Complexity** — Mått på hypotesklass rikedom via random label correlation.
**VC Dimension** — Största antal punkter klassificerbara godtyckligt; kapacitetsmått.
**Bias-Variance Decomposition** — Expected error = bias² + variance + irreducible noise.
**Maximum Likelihood Estimation** — MLE: välj θ som maximerar sannolikheten för observerad data.
**Maximum A Posteriori** — MAP: MLE med prior; ekvivalent med L2-regularisering för Gaussian prior.
**Bayesian Inference** — Uppdatera belief P(θ|D) med Bayes regel.
**Posterior Distribution** — P(θ|data) efter att ha sett evidens.
**Prior Distribution** — Belief om θ före data.
**Conjugate Prior** — Prior som ger samma familj som posterior (Beta-Binomial).
**Variational Inference** — Approximera posterior med enklare fördelning q(θ).
**ELBO** — Evidence Lower Bound; objektiv i variational autoencoders.
**Kullback-Leibler Divergence** — KL(P||Q): asymmetriskt mått på distributionskillnad.
**Cross-Entropy** — H(P,Q) = -Σ P log Q; classification loss och MDL.
**Mutual Information** — I(X;Y) mäter shared information mellan variabler.
**Entropy** — H(X) = -Σ p log p; osäkerhet i fördelning.
**Conditional Entropy** — H(X|Y); kvarvarande osäkerhet i X givet Y.
**Information Gain** — Entropiminskning vid split; används i decision trees.
**Softmax Function** — σ(z)_i = exp(z_i)/Σ exp(z_j); maps logits till simplex.
**Log-Sum-Exp Trick** — Numeriskt stabil beräkning av log(Σ exp).
**Sigmoid Function** — σ(x) = 1/(1+e^{-x}); binär klassificering och gating.
**ReLU Derivative** — 1 om x>0 else 0; subgradient vid x=0.
**Softplus** — Smooth approximation av ReLU: log(1+exp(x)).
**Gaussian Distribution** — N(μ,σ²); central i noise models och initiering.
**Multivariate Gaussian** — Vector generalization med mean μ och covariance Σ.
**Covariance Matrix** — Σ_ij = Cov(X_i, X_j); shape av joint distribution.
**Precision Matrix** — Inverse covariance; nolltermer indikerar conditional independence.
**Central Limit Theorem** — Summa av i.i.d. variabler → Gaussian; motiverar normal approx.
**Law of Large Numbers** — Sample mean konvergerar till expectation.
**Markov Chain** — Framtida beror endast på nuvarande state, inte historik.
**Stationary Distribution** — π där π P = π; långsiktig steady state av Markov kedja.
**Detailed Balance** — π(i)P(i→j) = π(j)P(j→i); sufficient för stationarity.
**Monte Carlo Estimation** — Approximera expectation med random samples.
**Importance Sampling** — Viktat sampling från q för att estimera under p.
**Markov Chain Monte Carlo** — MCMC: sample från komplex posterior (Metropolis-Hastings).
**Eigenvalue Decomposition** — A = QΛQ^T för symmetriska matriser; PCA-grund.
**Singular Value Decomposition** — SVD: A = UΣV^T; lågranksapproximation och stabilitet.
**Matrix Rank** — Dimension av kolumn-/radrymd; lågrank ≈ komprimerbar struktur.
**Condition Number** — κ(A) = σ_max/σ_min; mått på numerisk instabilitet.
**Positive Definite Matrix** — x^TAx > 0 ∀x≠0; covariance och Hessian i minimum.
**Dot Product** — x·y = Σ x_i y_i; geometric similarity och attention scores.
**Vector Norm** — L2: ||x||_2; L1 och L∞ för olika regularisering.
**Cosine Similarity Math** — x·y/(||x|| ||y||); vinkelbaserad likhet.
**Orthogonal Matrix** — Q^TQ = I; rotationer bevarar norm.
**Projection Matrix** — P = A(A^TA)^{-1}A^T; projicerar till kolumnrum.
**Lagrange Multipliers** — Constrained optimization via auxiliary multipliers.
**Karush-Kuhn-Tucker Conditions** — KKT: nödvändiga villkor för constrained optimum.
**Convex Optimization** — Globalt optimum för convex objektiv över convex set.
**Linear Programming** — Optimera linear objektiv under linear constraints.
**Quadratic Programming** — Quadratic objektiv; SVM dual form.
**Stochastic Process** — Random process över tid (Brownian motion, diffusion SDE).
**Brownian Motion** — Wiener process W_t; grund för SDE diffusion.
**Stochastic Differential Equation** — dX = f dt + g dW; score-based diffusion formulering.
**Itô's Lemma** — Chain rule för SDE; används i diffusion derivations.
**Fokker-Planck Equation** — PDE för tidsutveckling av densitet under SDE.
**Wasserstein Distance** — Earth mover's distance mellan fördelningar; GAN metric.
**Total Variation Distance** — max_A |P(A)-Q(A)|; starkt distributionsmått.
**Jensen's Inequality** — f(E[X]) ≤ E[f(X)] för convex f; motiverar variational bounds.
**Chebyshev's Inequality** — P(|X-μ|≥kσ) ≤ 1/k²; probabilistisk bound.
**Hoeffding Bound** — Exponential tail bound för summa av bounded variabler.
**Confidence Interval** — Intervall som täcker parameter med sannolikhet 1-α.
**Hypothesis Testing** — H0 vs H1 med p-value och signifikansnivå α.
**p-value** — Sannolikhet att observera data givet H0 sant.
**Type I Error** — False positive: reject true H0.
**Type II Error** — False negative: fail reject false H0.
**Statistical Power** — 1 - P(type II error); sannolikhet att detektera effekt.
**Bayes Factor** — Ratio of marginal likelihoods; evidence for H1 vs H0.
**Causal Inference** — Skilja correlation från causation med do-calculus, IV.
**Confounding Variable** — Z påverkar både X och Y; bias i naive correlation.
**Randomized Controlled Trial** — Gold standard för kausal effekt av intervention.
**Law of Total Expectation** — E[X] = E[E[X|Y]]; central för att marginalisera latent struktur.
**Law of Total Variance** — Var(X) = E[Var(X|Y)] + Var(E[X|Y]); dekomponerar osäkerhet.

## Forskningskoncept

**Ablation Study** — Systematiskt ta bort komponenter för att mäta deras bidrag.
**Baseline Comparison** — Jämför mot etablerad metod under identiska villkor.
**State of the Art** — SOTA: bästa kända resultat på given benchmark vid tidpunkt.
**Reproducibility Crisis** — Många resultat svåra att replikera p.g.a. saknad kod/data.
**Replication Study** — Oberoende team kör samma experiment för att verifiera.
**Pre-registration** — Publicera hypotes och metod före experiment för att undvika p-hacking.
**Peer Review** — Granskning av metod, novelty och validitet före publicering.
**Open Review** — Public reviews (OpenReview) för transparens.
**ArXiv Preprint** — Early dissemination före peer review.
**Conference vs Journal** — NeurIPS/ICML snabb cykel; journaler längre granskning.
**NeurIPS** — Premier ML-konferens; benchmark för cutting-edge forskning.
**ICML** — International Conference on Machine Learning.
**ICLR** — International Conference on Learning Representations.
**ACL EMNLP** — Top NLP-konferenser för språkteknologi.
**CVPR ICCV** — Ledande computer vision konferenser.
**Spotlight vs Oral** — Högt rankade papers får längre presentation.
**Best Paper Award** — Exceptionell novelty och impact enligt committee.
**Workshop Paper** — Mindre venue för tidiga idéer och niche topics.
**Technical Report** — Institutionell rapport utan full peer review.
**Novelty Claim** — Påstående om ny metod/insikt; måste stödjas av related work.
**Related Work Section** — Kontextualiserar bidrag mot prior art.
**Contribution Statement** — Explicit lista över paperets nya bidrag.
**Limitations Section** — Erkänner svagheter och scope-begränsningar.
**Broader Impact Statement** — Etiska och samhälleliga konsekvenser av forskning.
**Compute Reporting** — Dokumentera GPU-timmar och energi (NeurIPS krav).
**Hyperparameter Table** — Full lista HP för reproduktion.
**Appendix Details** — Arkitektur, prompts och extra experiments.
**Supplementary Material** — Extra figurer, proofs och kod-länkar.
**Anonymous Submission** — Double-blind review döljer författare.
**Single-Blind Review** — Reviewer anonym, författare känd.
**Author Response Period** — Rebuttal till reviewer-kommentarer.
**Meta-Review** — Area chair syntetiserar reviews till beslut.
**Accept Rate** — Andel submitted papers som accepteras.
**Citation Count** — Proxy för impact; laggad och fältberoende.
**h-index** — Forskare med h papers med ≥ h citations each.
**Impact Factor** — Journal metric; kontroversiell kvalitetsproxy.
**Seminal Paper** — Grundläggande arbete som definierar ett fält (Transformer, ResNet).
**Follow-Up Work** — Förbättringar och extensions av tidigare metod.
**Concurrent Discovery** — Oberoende teams publicerar liknande idéer samtidigt.
**Negative Results** — Rapportera att något inte fungerade; underskattat värde.
**Null Result Publication** — Publicera när hypotes inte bekräftas.
**Benchmark Overfitting** — Metod optimerad mot specifik test suite.
**Leaderboard Chasing** — Forskning driven av små metric gains utan insikt.
**Scaling Study** — Systematisk variation av modell/data/compute.
**Scaling Law Paper** — Empiriska potenslagar för loss vs scale.
**Emergent Phenomena Paper** — Förmågor som dyker upp vid skalning.
**Mechanistic Interpretability** — Förstå vilka circuits som implementerar beteende.
**Circuits Analysis** — Identifiera subgrafer i nätverk för specifika funktioner.
**Probing Classifiers** — Tränade på interna representationer för att testa encoding.
**Linear Probing** — Enkel linear classifier på frozen representations.
**Representation Similarity** — CKA/SVCCA jämför interna representationer mellan modeller.
**CKA Centered Kernel Alignment** — Mått på likhet mellan layer representations.
**Transfer Learning Study** — Pretrain på A, fine-tune på B; mät transfer.
**Zero-Shot Transfer** — Ingen fine-tune; direkt eval på ny uppgift.
**Few-Shot Learning Research** — Lär från få exempel via meta-learning eller prompting.
**Meta-Learning** — Learn to learn: MAML optimerar för snabb adaptation.
**MAML** — Model-Agnostic Meta-Learning; gradient-baserad meta-learning.
**In-Context Learning Research** — Studera ICL som implicit Bayesian inference.
**Inductive Bias Analysis** — Vilka strukturer arkitektur prioriterar.
**Universal Approximation Study** — Teoretiska kapacitetsresultat.
**Sample Complexity Analysis** — Antal exempel för ε-optimal generalization.
**Optimization Landscape** — Geometri av loss surface (saddle, minima).
**Loss Landscape Visualization** — 2D slices och mode connectivity.
**Mode Connectivity** — Olika minima kopplade via låg-loss paths.
**Lottery Ticket Hypothesis** — Små subnetworks kan tränas isolerat till samma prestanda.
**Double Descent Phenomenon** — Test error minskar igen efter interpolation threshold.
**Grokking** — Delayed generalization efter overfitting på algorithmic tasks.
**Benign Overfitting** — Interpolera träningsdata men generalisera ändå.
**Implicit Regularization** — SGD inducerar bias mot enkla lösningar utan explicit penalty.
**Neural Tangent Kernel** — NTK: infinite-width limit ger kernel regression.
**Lazy Training Regime** — Nätverk nära init; NTK approx gäller.
**Feature Learning Regime** — Aktiva representation changes; rikare än NTK.
**Scaling Hypothesis** — Testbar förutsägelse om beteende vid större scale.
**Bitter Lesson** — Sutton: general methods leveraging compute slår handcraft.
**Research Debt** — Accumulerad komplexitet och undocumented conventions.
**Benchmark Contamination Study** — Kvantifiera overlap träning/test.
**Data Provenance Research** — Spåra ursprung och licens av träningsdata.
**Synthetic Data Research** — Kvalitet och begränsningar av AI-genererad träningsdata.
**Multimodal Research Trend** — Unified models över text, bild, ljud.
**Agent Research Frontier** — Autonomous tool-using LLM systems.
**Reasoning Research** — Process supervision, verifiers, test-time compute.
**Open Science Movement** — Open weights, data och reproducibility standards.
**Collaborative Research** — Stora team och multi-institution papers.
**Industry Lab Publication** — Google, OpenAI, Meta driver mycket frontier research.
**Academic-Industry Gap** — Skillnad i compute access och publication strategy.
**Patent vs Publication** — IP skydd vs open dissemination tradeoff.
**Research Ethics Approval** — IRB för human subjects och sensitive data.
**Dual Submission Policy** — Regler mot samtidig submit till flera venues.
**Citation Ethics** — Korrekt attribut till prior work; undvik citation cartels.
**P-hacking Awareness** — Cherry-pick metrics, seeds eller subsets.

## NLP & Text

**Natural Language Processing** — NLP: automatisk förståelse, generering och transformation av text.
**Tokenization Strategy** — Val av subword-algoritm påverkar OOV, sekvenslängd och språk.
**WordPiece Tokenization** — Greedy subword merge använd i BERT; ## prefix för continuations.
**Unigram Language Model Tokenizer** — SentencePiece: probabilistiskt urval av subword vocab.
**Byte-Level BPE** — Opererar på bytes; robust för alla Unicode utan UNK.
**Part-of-Speech Tagging** — Tilldelar grammatisk kategori (noun, verb) per token.
**Named Entity Recognition** — NER: identifierar personer, organisationer, platser i text.
**Dependency Parsing** — Analyserar grammatiska beroenden mellan ord.
**Constituency Parsing** — Bygger phrase structure tree (NP, VP).
**Semantic Role Labeling** — Identifierar who did what to whom i meningar.
**Coreference Resolution** — Kopplar pronomen och uttryck till samma entitet.
**Word Sense Disambiguation** — Välj rätt betydelse av homonym i kontext.
**Machine Translation** — Automatisk översättning mellan språk.
**Neural Machine Translation** — Seq2seq transformer-baserad MT.
**Back-Translation** — Översätt monolingual data till synthetic parallel corpus.
**Text Summarization** — Komprimera dokument till kort sammanfattning.
**Extractive Summarization** — Välj viktiga meningar från källtext.
**Abstractive Summarization** — Generera ny formulering som sammanfattar innehåll.
**Question Answering** — Svara på frågor givet kontext eller öppen kunskap.
**Reading Comprehension** — Svara baserat på given passage (SQuAD-style).
**Open-Domain QA** — Hämta kunskap externt + generera svar.
**Closed-Book QA** — Svara enbart från parametrisk modellkunskap.
**Sentiment Analysis** — Klassificera positiv/negativ/neutral attityd.
**Aspect-Based Sentiment** — Sentiment per aspekt (mat, service) i recensioner.
**Text Classification** — Tilldela dokument en eller flera kategorier.
**Multi-Label Classification** — Flera etiketter samtidigt per dokument.
**Sequence Labeling** — Per-token labels (NER, POS).
**Sequence-to-Sequence** — Mappa input-sekvens till output-sekvens.
**Encoder-Decoder for NLP** — Transformer encoder-decoder för MT, summarization.
**Masked Language Model** — MLM: predice maskerade tokens; BERT pretraining.
**Causal Language Model** — Autoregressiv nästa-token prediction; GPT-style.
**Prefix LM** — Bidirectional prefix + autoregressiv suffix.
**UL2 Objective** — Mixture of denoisers med olika mask patterns.
**Span Corruption** — Maskera contiguous spans; T5 pretraining objective.
**Text Infilling** — Fyll i luckor i text som denoising objective.
**Denoising Autoencoder for Text** — Rekonstruera korrupt input; BART-style.
**Word Embeddings** — Dense vektorer för ord (Word2Vec, GloVe).
**Word2Vec Skip-gram** — Predicera context från center word.
**GloVe Embeddings** — Global co-occurrence statistics + weighted least squares.
**FastText** — Subword-aware embeddings; robust för morphology.
**Contextual Embeddings** — Representation beror på hela meningen (ELMo, BERT).
**ELMo** — Deep bidirectional LSTM embeddings från language model.
**Subword Regularization** — Sample olika segmenteringar under träning.
**Stemming and Lemmatization** — Reducera ord till stam/lemma för IR/NLP.
**Stop Word Removal** — Ta bort frekventa funktionsord; mindre viktigt med transformers.
**Bag of Words** — Räknar ordfrekvenser; enkel baseline.
**TF-IDF Vectorization** — Sparse representation för klassificering/IR.
**N-gram Language Model** — Predicerar nästa ord från n-1 föregångare.
**Perplexity in NLP** — Standard LM evaluation metric.
**Kneser-Ney Smoothing** — Avancerad smoothing för n-gram LM.
**Vocabulary Coverage** — Andel text som kan tokeniseras utan UNK.
**Out-of-Vocabulary Handling** — UNK token, byte fallback eller subword decomposition.
**Morphological Analysis** — Analysera morfem och böjningsformer.
**Multilingual NLP** — Modeller och pipelines för flera språk.
**Cross-Lingual Transfer** — Träna på ett språk, applicera på annat.
**Zero-Shot Cross-Lingual** — Ingen target language data vid träning.
**Language Identification** — Klassificera vilket språk text är på.
**Code-Switching** — Blandning av språk i samma yttrande.
**Transliteration** — Konvertera skript (latin ↔ Cyrillic).
**Text Normalization** — Expandera contractions, standardisera unicode.
**Truecasing** — Återställ korrekt versalisering.
**Sentence Segmentation** — Dela text i meningar.
**Word Alignment** — Mappa ord mellan käll- och målmening i MT.
**BLEU for MT** — N-gram precision mot reference translation.
**COMET Metric** — Neural metric korrelerad med human MT quality.
**Dialogue Systems** — Konversationsagent med context över turns.
**Task-Oriented Dialogue** — Slot-filling för booking, support.
**Open-Domain Dialogue** — Chitchat utan specifikt mål.
**Response Generation** — Generera relevant svar givet dialoghistorik.
**Dialogue State Tracking** — Spåra användarintent och slots över turns.
**Intent Classification** — Klassificera användarens avsikt i utterance.
**Slot Filling** — Extrahera parametrar (datum, plats) från user input.
**Information Extraction** — Strukturerad data från ostrukturerad text.
**Relation Extraction** — Identifiera relationer mellan entiteter.
**Event Extraction** — Hitta händelser och deltagare i text.
**Template Filling** — Populera fördefinierat schema från text.
**Semantic Parsing** — Mappa naturligt språk till logisk form/SQL.
**Text-to-SQL** — Generera SQL från naturlig språkfråga.
**SPARQL Generation** — Query knowledge graphs via naturligt språk.
**Constituency-to-Logic** — Kompositionell semantics för formal meaning.
**Discourse Parsing** — Analysera struktur över meningar (RST).
**Rhetorical Structure Theory** — RST: nucleus-satellite relationer i dokument.
**Coherence Modeling** — Bedöm om text hänger ihop logiskt.
**Lexical Substitution** — Hitta alternativa ord i samma kontext.
**Paraphrase Generation** — Omformulera mening med bibehållen betydelse.
**Paraphrase Detection** — Avgör om två meningar betyder samma sak.
**Semantic Textual Similarity** — STS: score grad av meningsslikhet.
**Natural Language Inference** — NLI: entailment, contradiction, neutral.
**Recognizing Textual Entailment** — RTE: klassisk NLI-uppgift.
**Stance Detection** — Klassificera pro/con/neutral mot target.

## Optimering & Träning

**Loss Function** — Skalar mål som minimeras under träning (CE, MSE, hinge).
**Cross-Entropy Loss** — Classification: -Σ y log ŷ; standard för LM och klassificering.
**Mean Squared Error** — MSE: regression loss (1/n)Σ(y-ŷ)².
**Huber Loss** — Robust regression: kvadratisk nära noll, linear för stora fel.
**Hinge Loss** — SVM loss: max(0, 1 - y·ŷ).
**Focal Loss for Classification** — Down-weight lätta exempel i class imbalance.
**Label Smoothing** — Mjuka targets (1-ε, ε/K) motverkar overconfidence.
**Class Weights in Loss** — Vikta sällsynta klasser högre i CE.
**Contrastive Loss** — Dra positiva par nära, push negativa bort (InfoNCE, triplet).
**Triplet Loss** — max(0, d(a,p) - d(a,n) + margin).
**InfoNCE Loss** — Noise contrastive estimation för representation learning.
**NT-Xent Loss** — Normalized temperature-scaled cross entropy i SimCLR.
**Denoising Loss** — Rekonstruktionsfel för maskerade/korrupta inputs.
**Perplexity as Training Metric** — Monitorera exp(CE) under LM träning.
**Gradient Descent Variants** — SGD, momentum, Adam familj.
**Mini-Batch SGD** — Stochastic gradient på delmängd per steg.
**Full-Batch Gradient Descent** — Gradient över hela dataset; sällan för stora data.
**Learning Rate Finder** — Sweep LR för att hitta optimalt intervall (LR range test).
**One-Cycle Learning Rate** — Öka sedan minska LR i ett pass; snabb träning.
**Cyclical Learning Rate** — Periodisk variation av LR.
**ReduceLROnPlateau** — Sänk LR när valideringsmetrik platear.
**Step Decay Schedule** — Multiplicera LR med faktor var N:e epoch.
**Linear Warmup Decay** — Warmup följt av linear decay till noll.
**Adam Optimizer** — Adaptive moment estimation med bias correction.
**AdamW Optimizer** — Decoupled weight decay; standard för transformers.
**Adam Beta Parameters** — β1 momentum, β2 RMS; typiskt 0.9 och 0.95-0.999.
**Epsilon in Adam** — Numerisk stabilisering i denominator (~1e-8).
**SGD with Momentum** — Klassisk optimizer med velocity term.
**Nesterov Accelerated Gradient** — Lookahead gradient evaluation.
**Adafactor Optimizer** — Memory-efficient Adam variant för stora modeller.
**8-bit Adam** — Kvantiserade optimizer states; sparar minne.
**Sophia Optimizer** — Second-order inspired; curvature-aware updates.
**Muon Optimizer** — Orthogonalized updates för hidden layer weights.
**Gradient Accumulation Steps** — Effective batch = micro_batch × accumulation.
**Global Batch Size** — Total batch över alla enheter per optimizer step.
**Micro Batch Size** — Per-GPU batch per forward pass.
**Gradient Clipping Norm** — Clip global gradient norm till max_värde.
**Gradient Clipping Value** — Element-wise clip till [-v, v].
**Weight Decay Regularization** — L2 penalty på vikter; decoupled i AdamW.
**L1 Regularization** — Sparsity-inducerande absolute value penalty.
**Dropout Regularization** — Stochastic neuron dropping under träning.
**Early Stopping** — Stoppa när validation loss inte förbättras.
**Patience in Early Stopping** — Antal epochs att vänta före stop.
**Checkpoint Averaging** — Medelvärde av sista k checkpoints (stochastic weight averaging).
**Stochastic Weight Averaging** — SWA: medelvärde längs training trajectory.
**Exponential Moving Average** — EMA av weights för stabilare eval model.
**Teacher EMA** — EMA teacher i self-supervised (MoCo, BYOL).
**Batch Normalization Momentum** — Running mean/var uppdateringshastighet.
**SyncBatchNorm** — Synkroniserad BN över GPU i distributed träning.
**Loss Scaling in FP16** — Multiplicera loss för att undvika gradient underflow.
**Dynamic Loss Scaling** — Automatisk justering av FP16 scale factor.
**Gradient Overflow Detection** — Skip step om Inf/NaN i gradient.
**Automatic Mixed Precision** — AMP: FP16 compute, FP32 master weights.
**BFloat16 Training** — BF16 mixed precision; bredare exponent range.
**TF32 on Ampere** — 19-bit matmul acceleration default på A100+.
**Distributed Data Parallel** — DDP: replikerad modell, synkroniserade gradienter.
**Horovod** — Ring allreduce för multi-GPU/multi-node träning.
**DeepSpeed ZeRO Stage 3** — Shardar parametrar, gradients och optimizer states.
**Optimizer State Sharding** — Dela Adam m/v över enheter.
**Pipeline Parallel Schedule** — 1F1B schedule minimerar pipeline bubble.
**Tensor Parallel All-Reduce** — Kommunikation efter column-parallel matmul.
**Communication Overlap** — Överlappa all-reduce med backward compute.
**Training Instability** — Loss spikes, NaN; ofta LR, precision eller init.
**Loss Spike Recovery** — Rollback checkpoint, sänk LR, skip bad batch.
**NaN Detection Hook** — Abort träning vid NaN i loss eller gradient.
**Weight Initialization Scale** — För stor init → explosion; för liten → stagnation.
**Residual Scaling Init** — Skala residual branch init för djupa nätverk.
**Learning Rate vs Batch Size Scaling** — Linear scaling rule: dubbla batch → dubbla LR.
**Square Root Scaling Rule** — Alternativ LR scaling för stora batch.
**Warmup Steps Calculation** — Typiskt 1-5% av totala training steps.
**Total Training Tokens** — Budget i tokens för LLM pretrain (Chinchilla).
**Chinchilla Token Budget** — ~20 tokens per parameter för compute-optimal.
**Over-Training** — Träna längre än compute-optimal; bättre inference efficiency.
**Data Mixture Ratio** — Viktning av subcorpora (web, code, books) i pretrain.
**Curriculum Sampling** — Öka svårare data under träning.
**Token-Level Loss Masking** — Exkludera padding och vissa tokens från loss.
**Sequence Length Curriculum** — Börja kort sekvens, öka gradvis.
**Flash Attention Training** — Minnes-effektiv exakt attention i träningsloop.
**Fused Optimizer Kernels** — Combined Adam update kernels för hastighet.
**torch.compile Training** — Graph compilation accelererar forward/backward.
**Profiling Training Step** — PyTorch profiler hittar CPU/GPU flaskhalsar.
**MFU Measurement** — Mät model FLOPs utilization under träning.
**Throughput Tokens per Second** — Träningshastighet i tokens/s per cluster.
**Time to Train Estimate** — Projicera wall-clock från throughput och token budget.
**Hyperparameter Sensitivity** — Robusthet mot små HP-ändringar.
**Seed Variance** — Resultatvariation över random seeds.
**Deterministic Training** — Cudnn deterministic, fix seeds; reproducerbarhet.
**Non-Deterministic Training** — Faster men slight variation mellan körningar.
**Cudnn Benchmark Mode** — Autotune conv algorithms; snabbare men icke-deterministisk.
**DataLoader Bottleneck Fix** — Öka workers, prefetch, pin_memory.

## Generativ AI & GANs

**Generative AI** — AI som skapar nytt innehåll (text, bild, ljud, video, kod).
**Generative Model** — Modellerar datadistribution P(x) eller P(x|y) för sampling.
**Discriminative Model** — Modellerar P(y|x) direkt; klassificering och regression.
**Generative Adversarial Network** — GAN: generator vs discriminator i minimax-spel.
**Generator Network** — G(z) mappar brus till synthetic data.
**Discriminator Network** — D(x) skiljer real vs fake; ger gradient till G.
**Minimax GAN Objective** — min_G max_D V(D,G) = E log D(x) + E log(1-D(G(z))).
**Non-Saturating GAN Loss** — Generator loss -log D(G(z)) för starkare gradient.
**Mode Collapse in GAN** — Generator producerar begränsad variation av samples.
**Training Instability in GAN** — Oscillerande loss; D eller G dominerar.
**Vanishing Gradient in GAN** — D för stark → G får ingen användbar gradient.
**Wasserstein GAN** — WGAN: använder Wasserstein distance med weight clipping/Lipschitz.
**WGAN-GP** — Gradient penalty istället för weight clipping för Lipschitz constraint.
**Spectral Normalization GAN** — Normalisera discriminator weights för stabilitet.
**Progressive GAN** — Gradvis öka upplösning under träning.
**StyleGAN** — Style-based generator med mapping network och adaptive instance norm.
**StyleGAN2** — Förbättrad arkitektur och path length regularization.
**StyleGAN3** — Alias-free generation för bättre animation och rotation.
**BigGAN** — Storskalig class-conditional GAN med spectral norm och big batch.
**Conditional GAN** — cGAN: conditionar G och D på label y.
**AC-GAN** — Auxiliary classifier GAN; auxiliary class loss på D.
**CycleGAN** — Unpaired image-to-image via cycle consistency loss.
**Pix2Pix** — Paired image-to-image med conditional GAN och L1 loss.
**SRGAN** — Super-resolution GAN med perceptual loss.
**ProGAN** — Progressive growing för högkvalitativa ansikten.
**DCGAN** — Deep Convolutional GAN med riktlinjer för stabil arkitektur.
**Latent Space Interpolation** — Linjär interpolation z1→z2 i generator latent space.
**Latent Vector z** — Brusvektor som input till generator.
**GAN Inversion** — Hitta z som rekonstruerar given bild genom G.
**Perceptual Loss** — L2 på feature maps från pretrained network (VGG).
**Feature Matching Loss** — Match intermediate D features real vs fake.
**Historical Averaging GAN** — Regularisera G mot tidigare parametrar.
**Unrolled GAN** — Lookahead D optimization för stabilare G gradient.
**Self-Attention GAN** — SAGAN: attention i generator och discriminator.
**Projection Discriminator** — Class embedding projiceras på features i D.
**R1 Gradient Penalty** — Regularisera D gradient norm på real data.
**Consistency Regularization GAN** — CR-GAN för semi-supervised.
**Energy-Based Model** — EBM: assign low energy to data, high to other.
**Normalizing Flow** — Invertible transform med exact likelihood (RealNVP, Glow).
**RealNVP** — Affine coupling layers för bijective transform.
**Glow** — 1x1 conv + affine coupling; efficient flow model.
**Variational Autoencoder** — VAE: encoder q(z|x), decoder p(x|z), maximize ELBO.
**Reparameterization Trick** — z = μ + σ·ε för backprop genom stochastic node.
**KL Divergence in VAE** — Regularisera latent mot prior N(0,I).
**β-VAE** — Viktad KL för disentangled representations.
**Vector Quantized VAE** — VQ-VAE: discrete latent codes för generering.
**Codebook in VQ-VAE** — Learned embedding table för quantized latents.
**Autoregressive Generative Model** — Factorize P(x) som produkt av conditionals.
**PixelCNN** — Autoregressiv generering pixel för pixel.
**WaveNet** — Autoregressiv råljudsgenerering med dilated conv.
**Transformer LM as Generator** — GPT genererar text autoregressivt.
**Diffusion as Generative Model** — Alternativ till GAN/VAE med iterativ denoising.
**Flow Matching Generative** — Continuous normalizing flows utan simulering.
**Consistency Model Generation** — Few-step generering från diffusion family.
**Score-Based Generative Model** — Lär score function för sampling via SDE.
**Implicit Generative Model** — Sample utan explicit density (GAN).
**Explicit Likelihood Model** — Normalizing flows, autoregressive med tractable P(x).
**Evaluating Generative Models** — FID, IS, precision/recall for generative models.
**Precision and Recall for GANs** — Quality vs diversity decomposition.
**Fréchet Inception Distance** — FID: distance mellan feature Gaussians.
**Inception Score** — IS: klassificerbarhet och diversity av generated images.
**Kernel Inception Distance** — KID: unbiased alternative till FID.
**Neural Audio Codec** — EnCodec/DAC: komprimerar ljud till discrete tokens för gen.
**Neural Vocoder** — WaveGlow, HiFi-GAN: waveform från mel-spectrogram.
**HiFi-GAN** — High-fidelity GAN vocoder för TTS.
**Mel Spectrogram** — Tids-frekvensrepresentation för ljudmodeller.
**Text-to-Music Generation** — Generera musik från text eller strukturerad prompt.
**Text-to-Video Generation** — Diffusion/transformer genererar videosekvenser.
**Video GAN** — Generera video frames med temporal consistency.
**3D Generative Model** — Generera meshes, NeRF eller Gaussians.
**DreamFusion** — Text-to-3D via 2D diffusion distillation och NeRF optimization.
**Generative Fill** — Inpainting i bildredigering med generativ modell.
**ControlNet for Generation** — Spatial control i generativ pipeline.
**LoRA for Style Generation** — Personlig stil/koncept utan full retrain.
**Prompt Engineering for T2I** — Formulera effektiva bildprompter.
**Negative Prompt Engineering** — Undvik oönskade element i generering.
**Seed Control in Generation** — Fix noise seed för reproducerbara bilder.
**Batch Generation** — Generera många samples parallellt.
**Classifier Guidance** — Gradient från classifier styr diffusion sampling.
**CFG in Diffusion** — Classifier-free guidance utan separat classifier.
**Generative Model Safety** — NSFW filter och watermarking i generativ pipeline.
**Synthetic Data Generation** — Generera träningsdata med generativa modeller.
**Data Augmentation via Generation** — GAN/diffusion skapar extra träningsexempel.
**Privacy-Preserving Generation** — Synthetic data utan att memorerar PII.
**Membership Inference on Generative Models** — Kan attacker avgöra om sample var i träningsdata?
**Model Collapse from Synthetic Data** — Träna på AI-genererad data degraderar framtida modeller.
**Human Evaluation of Generative Output** — A/B, Elo, Likert för perceptuell kvalitet.
**Turing Test for Generative AI** — Kan output skiljas från mänskligt?
**Creative AI Applications** — Konst, design, musik med generativa modeller.
**Generative AI Copyright** — Vem äger AI-genererat verk?

Total terms: 2000
