"""
Test dataset for evaluating the MediChat RAG chatbot.
Each entry has:
  - question: the user query
  - ground_truth: a reference answer based on known medical facts (from WHO sources)
  - expected_sources: list of source IDs that should appear in retrieval results
"""

TEST_QUESTIONS = [
    # 1
    {
        "question": "What are the main symptoms of asthma?",
        "ground_truth": (
            "Common symptoms of asthma include wheezing, shortness of breath, "
            "chest tightness, and coughing. Symptoms can vary in severity and may "
            "worsen at night or during exercise."
        ),
        "expected_sources": ["asthma"],
    },
    # 2
    {
        "question": "What causes diabetes?",
        "ground_truth": (
            "Type 1 diabetes is caused by an autoimmune reaction that destroys "
            "insulin-producing beta cells in the pancreas. Type 2 diabetes is caused "
            "by the body's ineffective use of insulin, often linked to excess body "
            "weight, physical inactivity, and genetic factors."
        ),
        "expected_sources": ["diabetes"],
    },
    # 3
    {
        "question": "How is hypertension diagnosed?",
        "ground_truth": (
            "Hypertension is diagnosed by measuring blood pressure. A person is "
            "diagnosed with hypertension when their systolic blood pressure is 140 mmHg "
            "or higher and/or their diastolic blood pressure is 90 mmHg or higher on "
            "two different days."
        ),
        "expected_sources": ["hypertension"],
    },
    # 4
    {
        "question": "What are the risk factors for lung cancer?",
        "ground_truth": (
            "The main risk factors for lung cancer include tobacco smoking, exposure "
            "to secondhand smoke, occupational exposure to carcinogens like asbestos "
            "and radon, air pollution, and a family history of lung cancer."
        ),
        "expected_sources": ["lung_cancer"],
    },
    # 5
    {
        "question": "What is avascular necrosis?",
        "ground_truth": (
            "Avascular necrosis (osteonecrosis) is a condition where bone tissue dies "
            "due to a lack of blood supply. It can lead to tiny breaks in the bone and "
            "eventual collapse. It most commonly affects the hip joint."
        ),
        "expected_sources": ["avascular_necrosis"],
    },
    # 6
    {
        "question": "How can malaria be prevented?",
        "ground_truth": (
            "Malaria can be prevented through insecticide-treated mosquito nets, indoor "
            "residual spraying, antimalarial medications for travellers, and preventive "
            "chemotherapy for high-risk groups."
        ),
        "expected_sources": ["malaria"],
    },
    # 7
    {
        "question": "What are the symptoms of depression?",
        "ground_truth": (
            "Symptoms of depression include persistent sad or empty mood, loss of "
            "interest in activities, changes in appetite and sleep, fatigue, difficulty "
            "concentrating, feelings of worthlessness, and thoughts of death or suicide."
        ),
        "expected_sources": ["depression"],
    },
    # 8
    {
        "question": "How is tuberculosis transmitted?",
        "ground_truth": (
            "Tuberculosis is spread person to person through the air when people with "
            "active pulmonary TB cough, sneeze, speak, or spit, propelling bacteria "
            "into the air."
        ),
        "expected_sources": ["tuberculosis"],
    },
    # 9
    {
        "question": "What treatments are available for HIV/AIDS?",
        "ground_truth": (
            "HIV is treated with antiretroviral therapy (ART), a combination of HIV "
            "medicines taken daily. ART does not cure HIV but helps people live longer, "
            "healthier lives and reduces transmission risk."
        ),
        "expected_sources": ["hiv_aids"],
    },
    # 10
    {
        "question": "What is the recommended diet for someone with cardiovascular disease?",
        "ground_truth": (
            "A heart-healthy diet includes fruits, vegetables, whole grains, lean "
            "proteins, and healthy fats, while limiting salt, sugar, saturated fat, "
            "and trans fat intake."
        ),
        "expected_sources": ["cardiovascular_diseases_(cvds)"],
    },
    # 11
    {
        "question": "What are the symptoms of cholera?",
        "ground_truth": (
            "Cholera can cause profuse watery diarrhoea (rice-water stools), vomiting, "
            "and rapid dehydration. Severe cases can lead to death within hours if untreated."
        ),
        "expected_sources": ["cholera"],
    },
    # 12
    {
        "question": "How is dengue fever transmitted?",
        "ground_truth": (
            "Dengue is transmitted through the bites of infected female Aedes mosquitoes, "
            "primarily Aedes aegypti. The mosquitoes become infected when they bite a "
            "person with dengue virus in their blood."
        ),
        "expected_sources": ["dengue_and_severe_dengue"],
    },
    # 13
    {
        "question": "What is COPD and what causes it?",
        "ground_truth": (
            "Chronic obstructive pulmonary disease (COPD) is a chronic lung disease "
            "that causes obstructed airflow. The main cause is tobacco smoking, with "
            "other risk factors including air pollution and occupational dusts."
        ),
        "expected_sources": ["chronic_obstructive_pulmonary_disease_(copd)"],
    },
    # 14
    {
        "question": "What are the warning signs of a stroke?",
        "ground_truth": (
            "Warning signs of stroke include sudden numbness or weakness of the face, "
            "arm, or leg (especially on one side), confusion, trouble speaking, vision "
            "problems, difficulty walking, and severe headache."
        ),
        "expected_sources": ["stroke"],
    },
    # 15
    {
        "question": "How is hepatitis B transmitted?",
        "ground_truth": (
            "Hepatitis B is transmitted through contact with infected blood and body "
            "fluids, including from mother to child at birth, through unsafe injections, "
            "sexual contact, and needle sharing."
        ),
        "expected_sources": ["hepatitis_b"],
    },
    # 16
    {
        "question": "What are the symptoms of measles?",
        "ground_truth": (
            "Measles symptoms include high fever, cough, runny nose, red watery eyes, "
            "and a rash of flat red spots that appears on the face and spreads to the "
            "rest of the body."
        ),
        "expected_sources": ["measles"],
    },
    # 17
    {
        "question": "How can obesity be prevented?",
        "ground_truth": (
            "Obesity can be prevented by limiting energy intake from fats and sugars, "
            "increasing consumption of fruits, vegetables, and whole grains, and "
            "engaging in regular physical activity."
        ),
        "expected_sources": ["obesity_and_overweight"],
    },
    # 18
    {
        "question": "What is epilepsy and how is it treated?",
        "ground_truth": (
            "Epilepsy is a chronic brain disorder characterized by recurrent seizures. "
            "It can be treated with anti-seizure medications, and up to 70% of people "
            "with epilepsy could become seizure-free with appropriate treatment."
        ),
        "expected_sources": ["epilepsy"],
    },
    # 19
    {
        "question": "What causes rabies and how can it be prevented?",
        "ground_truth": (
            "Rabies is caused by a virus transmitted through the saliva of infected "
            "animals, usually via bites. It can be prevented through vaccination of "
            "dogs and post-exposure prophylaxis after animal bites."
        ),
        "expected_sources": ["rabies"],
    },
    # 20
    {
        "question": "What is dementia and what are its early signs?",
        "ground_truth": (
            "Dementia is a syndrome involving deterioration of memory, thinking, "
            "behaviour, and the ability to perform everyday activities. Early signs "
            "include forgetfulness, losing track of time, and becoming lost in "
            "familiar places."
        ),
        "expected_sources": ["dementia"],
    },
    # 21
    {
        "question": "How does influenza spread?",
        "ground_truth": (
            "Seasonal influenza spreads easily through respiratory droplets and small "
            "particles when infected people cough, sneeze, or talk. It can also spread "
            "by touching contaminated surfaces."
        ),
        "expected_sources": ["influenza_(seasonal)"],
    },
    # 22
    {
        "question": "What are the health effects of lead poisoning?",
        "ground_truth": (
            "Lead poisoning can affect nearly every organ system. In children it causes "
            "developmental delays, learning difficulties, and behavioural problems. "
            "In adults it can cause hypertension, kidney damage, and reproductive problems."
        ),
        "expected_sources": ["lead_poisoning_and_health"],
    },
    # 23
    {
        "question": "What is cervical cancer and how can it be prevented?",
        "ground_truth": (
            "Cervical cancer develops in the cervix and is almost always caused by "
            "human papillomavirus (HPV) infection. It can be prevented through HPV "
            "vaccination, regular screening, and treatment of pre-cancerous lesions."
        ),
        "expected_sources": ["cervical_cancer"],
    },
    # 24
    {
        "question": "What are the symptoms of hepatitis C?",
        "ground_truth": (
            "Most people with hepatitis C are asymptomatic for years. When symptoms "
            "appear they may include fever, fatigue, decreased appetite, nausea, "
            "abdominal pain, dark urine, and jaundice."
        ),
        "expected_sources": ["hepatitis_c"],
    },
    # 25
    {
        "question": "How is meningitis treated?",
        "ground_truth": (
            "Bacterial meningitis is treated with antibiotics, which should be started "
            "as soon as possible. Supportive care and corticosteroids may also be used. "
            "Viral meningitis is usually self-limiting."
        ),
        "expected_sources": ["meningitis"],
    },
    # 26
    {
        "question": "What is schizophrenia?",
        "ground_truth": (
            "Schizophrenia is a severe mental disorder affecting thinking, perception, "
            "emotions, language, sense of self, and behaviour. Common symptoms include "
            "hallucinations, delusions, and disorganized thinking."
        ),
        "expected_sources": ["schizophrenia"],
    },
    # 27
    {
        "question": "What are the complications of diabetes?",
        "ground_truth": (
            "Diabetes complications include cardiovascular disease, nerve damage "
            "(neuropathy), kidney damage (nephropathy), eye damage (retinopathy), "
            "foot damage, and increased risk of infections."
        ),
        "expected_sources": ["diabetes"],
    },
    # 28
    {
        "question": "How is tetanus prevented?",
        "ground_truth": (
            "Tetanus is prevented through vaccination with tetanus toxoid-containing "
            "vaccines. Proper wound care and cleaning also reduces the risk of "
            "tetanus infection."
        ),
        "expected_sources": ["tetanus"],
    },
    # 29
    {
        "question": "What are the effects of air pollution on health?",
        "ground_truth": (
            "Air pollution increases the risk of respiratory infections, heart disease, "
            "stroke, lung cancer, and COPD. Fine particulate matter (PM2.5) penetrates "
            "deep into the lungs and bloodstream."
        ),
        "expected_sources": ["ambient_(outdoor)_air_quality_and_health"],
    },
    # 30
    {
        "question": "What is bipolar disorder?",
        "ground_truth": (
            "Bipolar disorder is a mental health condition characterized by extreme "
            "mood swings including emotional highs (mania or hypomania) and lows "
            "(depression). Episodes can affect sleep, energy, activity, and judgement."
        ),
        "expected_sources": ["bipolar_disorder"],
    },
    # 31
    {
        "question": "How can drowning be prevented?",
        "ground_truth": (
            "Drowning prevention strategies include installing barriers to control "
            "access to water, teaching swimming and water safety, providing safe "
            "places for children, and training bystanders in rescue and resuscitation."
        ),
        "expected_sources": ["drowning"],
    },
    # 32
    {
        "question": "What are the risk factors for breast cancer?",
        "ground_truth": (
            "Risk factors for breast cancer include increasing age, family history, "
            "genetic mutations (BRCA1/BRCA2), early menarche, late menopause, obesity, "
            "alcohol consumption, and hormone replacement therapy."
        ),
        "expected_sources": ["breast_cancer"],
    },
    # 33
    {
        "question": "What is sepsis and what causes it?",
        "ground_truth": (
            "Sepsis is a life-threatening organ dysfunction caused by a dysregulated "
            "host response to infection. It can be caused by bacterial, viral, fungal, "
            "or parasitic infections and requires urgent medical treatment."
        ),
        "expected_sources": ["sepsis"],
    },
    # 34
    {
        "question": "How is polio transmitted?",
        "ground_truth": (
            "Polio (poliomyelitis) is transmitted person to person mainly through the "
            "faecal-oral route or, less frequently, through contaminated water or food. "
            "The virus multiplies in the intestine."
        ),
        "expected_sources": ["poliomyelitis"],
    },
    # 35
    {
        "question": "What are the symptoms of Parkinson disease?",
        "ground_truth": (
            "Parkinson disease symptoms include tremor, slowness of movement (bradykinesia), "
            "limb rigidity, and gait and balance problems. Non-motor symptoms include "
            "cognitive impairment, sleep disorders, and depression."
        ),
        "expected_sources": ["parkinson_disease"],
    },
    # 36
    {
        "question": "What is endometriosis?",
        "ground_truth": (
            "Endometriosis is a condition where tissue similar to the uterine lining "
            "grows outside the uterus. It causes chronic pelvic pain, painful periods, "
            "pain during intercourse, and can lead to infertility."
        ),
        "expected_sources": ["endometriosis"],
    },
    # 37
    {
        "question": "How do snakebites affect the body?",
        "ground_truth": (
            "Venomous snakebites can cause paralysis, bleeding disorders, kidney "
            "failure, tissue destruction, and death. Effects depend on the species "
            "and amount of venom injected."
        ),
        "expected_sources": ["snakebite_envenoming"],
    },
    # 38
    {
        "question": "What is leishmaniasis?",
        "ground_truth": (
            "Leishmaniasis is a parasitic disease transmitted by the bite of infected "
            "sandflies. The three main forms are visceral (kala-azar), cutaneous, and "
            "mucocutaneous leishmaniasis."
        ),
        "expected_sources": ["leishmaniasis"],
    },
    # 39
    {
        "question": "What causes food poisoning from salmonella?",
        "ground_truth": (
            "Salmonella food poisoning is caused by eating food contaminated with "
            "Salmonella bacteria, commonly found in raw eggs, undercooked poultry, "
            "meat, and unpasteurized milk."
        ),
        "expected_sources": ["salmonella_(non_typhoidal)"],
    },
    # 40
    {
        "question": "What is the treatment for leprosy?",
        "ground_truth": (
            "Leprosy is treated with multidrug therapy (MDT), a combination of "
            "dapsone, rifampicin, and clofazimine. MDT cures the disease and prevents "
            "disability when started early."
        ),
        "expected_sources": ["leprosy"],
    },
    # 41
    {
        "question": "What are the health risks of tobacco use?",
        "ground_truth": (
            "Tobacco use causes lung cancer, COPD, cardiovascular disease, stroke, "
            "and many other cancers. It is one of the leading preventable causes of "
            "death worldwide."
        ),
        "expected_sources": ["tobacco"],
    },
    # 42
    {
        "question": "What is preterm birth and what are its risk factors?",
        "ground_truth": (
            "Preterm birth is birth before 37 completed weeks of gestation. Risk factors "
            "include previous preterm birth, multiple pregnancies, infections, chronic "
            "conditions like diabetes and hypertension, and maternal age."
        ),
        "expected_sources": ["preterm_birth"],
    },
    # 43
    {
        "question": "How is yellow fever prevented?",
        "ground_truth": (
            "Yellow fever is prevented by vaccination. A single dose of the yellow "
            "fever vaccine provides lifelong immunity. Mosquito control measures also "
            "help reduce transmission."
        ),
        "expected_sources": ["yellow_fever"],
    },
    # 44
    {
        "question": "What is osteoarthritis?",
        "ground_truth": (
            "Osteoarthritis is the most common form of arthritis, causing joint pain, "
            "stiffness, and reduced mobility. It occurs when the protective cartilage "
            "that cushions the ends of bones wears down over time."
        ),
        "expected_sources": ["osteoarthritis"],
    },
    # 45
    {
        "question": "What are the symptoms of Ebola?",
        "ground_truth": (
            "Ebola symptoms include sudden onset of fever, fatigue, muscle pain, "
            "headache, and sore throat, followed by vomiting, diarrhoea, rash, "
            "and in some cases internal and external bleeding."
        ),
        "expected_sources": ["ebola_disease"],
    },
    # 46
    {
        "question": "What is the impact of alcohol on health?",
        "ground_truth": (
            "Harmful alcohol use causes liver disease, cardiovascular disease, cancers, "
            "mental health problems, and injuries. It is a leading risk factor for "
            "premature death and disability worldwide."
        ),
        "expected_sources": ["alcohol"],
    },
    # 47
    {
        "question": "How does schistosomiasis spread?",
        "ground_truth": (
            "Schistosomiasis is spread through contact with freshwater contaminated "
            "with parasitic worms. People become infected when larval forms of the "
            "parasite penetrate the skin during contact with infested water."
        ),
        "expected_sources": ["schistosomiasis"],
    },
    # 48
    {
        "question": "What are the signs of anxiety disorders?",
        "ground_truth": (
            "Anxiety disorders involve excessive fear and worry that is out of "
            "proportion to the situation. Symptoms include restlessness, fatigue, "
            "difficulty concentrating, irritability, muscle tension, and sleep problems."
        ),
        "expected_sources": ["anxiety_disorders"],
    },
    # 49
    {
        "question": "What is palliative care?",
        "ground_truth": (
            "Palliative care is an approach that improves quality of life of patients "
            "and families facing life-threatening illness, through prevention and relief "
            "of suffering by means of early identification, assessment, and treatment "
            "of pain and other problems."
        ),
        "expected_sources": ["palliative_care"],
    },
    # 50
    {
        "question": "How are burns treated?",
        "ground_truth": (
            "Burns treatment depends on severity. First aid includes cooling the burn "
            "with running water. Severe burns require medical care including wound "
            "cleaning, dressings, pain management, fluid resuscitation, and sometimes "
            "skin grafting."
        ),
        "expected_sources": ["burns"],
    },
]
