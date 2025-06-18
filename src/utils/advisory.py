# -*- coding: utf-8 -*-
"""
Complete Multilingual Agricultural Advisory Script for the Indian Landscape.

Covers all 38 classes from the KrishiRakshak dataset. Provides actionable,
verifiable advice with a focus on low-cost organic and standard chemical
treatments relevant to Indian farmers.

Languages: English (en), Hindi (hi), Marathi (mr)
Source Verification: Advisories are based on recommendations from Indian
Council of Agricultural Research (ICAR), State Agricultural Universities (SAUs),
and Krishi Vigyan Kendras (KVKs).
"""

# --- 1. Advisory Content Database (Base Language: English) ---
# Contains detailed advisories for all 26 non-healthy conditions in the dataset.

ADVISORY_DATA = {
    # == Apple Diseases ==
    'Apple___Apple_scab': {
        'common_name': "Apple Scab / सेब का स्कैब / सफरचंदावरील स्कॅब (खरुज)",
        'symptoms': "Velvety, olive-green spots on leaves, which later turn black and cause the leaf to curl. On fruit, it causes dark, corky, scabby spots, making the fruit deformed and unmarketable.",
        'prevention': ["Prune trees annually during dormancy to ensure good air circulation.", "Rake and burn all fallen leaves in late autumn to destroy the overwintering fungus.", "Choose scab-resistant varieties if planting a new orchard."],
        'organic_management': ["Dormant Spray: Spray lime-sulfur solution on dormant trees before bud-break.", "Foliar Spray: After leaves emerge, spray a solution of Bordeaux mixture or wettable sulfur."],
        'chemical_management': ["Spray Schedule: A strict spray schedule is critical.", "Green Tip Stage: Spray Mancozeb @ 3 grams/liter of water.", "Petal Fall Stage: Spray a systemic fungicide like Myclobutanil or Difenoconazole @ 0.5 grams/liter of water.", "**Disclaimer**: Consult local horticultural experts for the precise spray schedule in your region. Always read and follow label instructions."],
        'source': "Based on recommendations from SKUAST, CITH (ICAR) & National Horticulture Board (NHB)."
    },
    'Apple___Black_rot': {
        'common_name': "Apple Black Rot / सेब का काला सड़न / सफरचंदावरील काळी सड",
        'symptoms': "Causes 'frog-eye' leaf spots (purple specks enlarging to brown circles), limb cankers, and a firm, brown-to-black rot on the fruit, often starting at the blossom end.",
        'prevention': ["Prune out dead or cankered branches.", "Remove mummified fruit from the tree and the ground.", "Maintain tree health through proper watering and fertilization."],
        'organic_management': ["Surgical removal of cankers from large branches and application of Bordeaux paste.", "Spray copper-based fungicides during the dormant season."],
        'chemical_management': ["Apply fungicides like Captan or Zineb during the growing season, especially after hail or other injuries.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR & State Horticultural Departments."
    },
    'Apple___Cedar_apple_rust': {
        'common_name': "Cedar Apple Rust / देवदार-सेब रस्ट / देवदार-सफरचंद गंज",
        'symptoms': "Bright yellow-orange spots on leaves and fruit. On the underside of leaves, small cup-like structures may form. Requires a nearby juniper or cedar tree to complete its lifecycle.",
        'prevention': ["Remove nearby juniper and cedar trees if possible (up to a 1-2 km radius).", "Plant rust-resistant apple varieties."],
        'organic_management': ["Spraying with wettable sulfur can provide some protection if applied before infection.", "Kaolin clay sprays can form a protective barrier on leaves."],
        'chemical_management': ["Apply preventative fungicides (e.g., Myclobutanil, Mancozeb) starting at bud break.", "Once spots are visible, treatment is less effective for the current season.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on general plant pathology recommendations adapted for India."
    },

    # == Cherry Disease ==
    'Cherry_(including_sour)___Powdery_mildew': {
        'common_name': "Cherry Powdery Mildew / चेरी का पाउडरी मिल्ड्यू / चेरीवरील भुरी रोग",
        'symptoms': "White, powdery patches on leaves and shoots. Infected leaves may distort or curl. Can also affect fruit, causing russeting.",
        'prevention': ["Ensure good air circulation through proper pruning.", "Avoid excessive nitrogen fertilizer, which encourages susceptible new growth."],
        'organic_management': ["Spray horticultural oils or neem oil at the first sign of disease.", "Potassium bicarbonate solution (10g in 1 liter water) can be an effective contact fungicide."],
        'chemical_management': ["Apply fungicides such as wettable sulfur at the first sign of disease.", "Systemic fungicides like Myclobutanil can be used in severe cases.", "**Disclaimer**: Do not apply sulfur when temperatures are high (above 30°C). Always follow label instructions."],
        'source': "Based on recommendations from ICAR & State Horticultural Departments."
    },

    # == Corn (Maize) Diseases ==
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'common_name': "Maize Gray Leaf Spot / मक्का का ग्रे लीफ स्पॉट / मक्यावरील ग्रे लीफ स्पॉट (करपा)",
        'symptoms': "Starts as small, water-soaked spots that evolve into long, narrow, rectangular brown or gray lesions, typically parallel to leaf veins.",
        'prevention': ["Use resistant corn hybrids.", "Rotate crops with non-grass species like soybean or cotton.", "Manage crop residue by tilling to reduce fungal survival."],
        'organic_management': ["Promote soil health with compost and organic manure to build plant immunity.", "A preventative spray of neem oil (10 ml/liter water) can help reduce fungal spore germination."],
        'chemical_management': ["Fungicide application (e.g., Pyraclostrobin, Azoxystrobin) may be necessary if the disease appears before tasseling.", "**Disclaimer**: Economic viability of spraying should be considered. Always follow label instructions."],
        'source': "Based on recommendations from Indian Institute of Maize Research (ICAR-IIMR)."
    },
    'Corn_(maize)___Common_rust_': {
        'common_name': "Maize Common Rust / मक्का का सामान्य रस्ट / मक्यावरील सामान्य तांबेरा",
        'symptoms': "Small, cinnamon-brown, powdery pustules on both upper and lower leaf surfaces. Pustules are oval or elongated.",
        'prevention': ["Planting resistant hybrids is the most effective method.", "Early planting can sometimes help the crop mature before rust becomes severe."],
        'organic_management': ["Generally not considered severe enough for extensive organic intervention in field corn.", "Maintaining good plant nutrition can help tolerate the infection."],
        'chemical_management': ["Generally not economically damaging enough to require fungicide treatment in most field corn.", "For sweet corn or seed production, fungicides like Propiconazole can be used if rust is severe.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from Indian Institute of Maize Research (ICAR-IIMR)."
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'common_name': "Maize Northern Leaf Blight / मक्का का उत्तरी पत्ती झुलसा / मक्यावरील नॉर्दन लीफ ब्लाईट (करपा)",
        'symptoms': "Large, cigar-shaped, grayish-green to tan lesions on the leaves, typically 1 to 6 inches long.",
        'prevention': ["Plant resistant hybrids recommended for your region.", "Crop rotation and tillage help reduce inoculum."],
        'organic_management': ["Application of *Trichoderma viride* to the soil can help improve plant defense.", "Maintaining balanced soil fertility helps the plant withstand infection."],
        'chemical_management': ["Fungicides like Propiconazole or Azoxystrobin can be effective if applied early, when lesions first appear on lower leaves.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from Indian Institute of Maize Research (ICAR-IIMR)."
    },

    # == Grape Diseases ==
    'Grape___Black_rot': {
        'common_name': "Grape Black Rot / अंगूर का काला सड़न / द्राक्षावरील काळी सड (ब्लॅक रॉट)",
        'symptoms': "Small, brown circular spots on leaves that develop black borders. On fruit, it starts as a pale spot, then the entire berry rots, turns black, and shrivels into a hard 'mummy'.",
        'prevention': ["Prune vines in the dormant season to improve air circulation.", "Remove and destroy mummified berries and infected canes.", "Choose a sunny, well-drained planting site."],
        'organic_management': ["Apply dormant sprays of lime-sulfur.", "Regular sprays of Bordeaux mixture during the growing season are effective.", "Bagging individual fruit clusters can provide physical protection."],
        'chemical_management': ["Apply preventative fungicides (e.g., Mancozeb, Captan) from early shoot growth until the berries begin to ripen.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from National Research Centre for Grapes (ICAR-NRCG)."
    },
    'Grape___Esca_(Black_Measles)': {
        'common_name': "Grape Esca (Black Measles) / अंगूर का एस्का (ब्लैक मीसल्स) / द्राक्षावरील एस्का (ब्लॅक मिझल्स)",
        'symptoms': "Can be chronic (small, dark spots on berries, 'tiger-stripe' patterns on leaves) or acute (sudden wilting and death of the entire vine).",
        'prevention': ["Protect pruning wounds, especially large ones, with a sealant like Bordeaux paste.", "Avoid pruning in the rain.", "Remove and destroy dead or diseased vines immediately."],
        'organic_management': ["The primary organic approach is prevention through careful pruning and sanitation.", "*Trichoderma*-based pastes can be applied to large pruning wounds to prevent fungal entry."],
        'chemical_management': ["There is no effective chemical cure. Management relies on good sanitation and preventative pruning practices to stop the spread of the wood-decaying fungi.", "**Disclaimer**: Prevention is the only viable strategy."],
        'source': "Based on recommendations from National Research Centre for Grapes (ICAR-NRCG)."
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'common_name': "Grape Leaf Blight / अंगूर का पत्ती झुलसा / द्राक्षावरील पानावरील करपा",
        'symptoms': "Irregular, dark reddish-brown to black spots on leaves, which may cause leaves to drop prematurely. The underside of the spots may appear dark and sooty.",
        'prevention': ["Good sanitation, including removing fallen leaves.", "Proper pruning for good air circulation to keep foliage dry."],
        'organic_management': ["Sprays of Bordeaux mixture or copper-based fungicides provide good control.", "Neem oil sprays can help reduce the spread."],
        'chemical_management': ["Fungicide sprays used for other grape diseases like Black Rot (e.g., Mancozeb, Chlorothalonil) will typically control Leaf Blight as well.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from National Research Centre for Grapes (ICAR-NRCG)."
    },

    # == Orange Disease ==
    'Orange___Haunglongbing_(Citrus_greening)': {
        'common_name': "Citrus Greening (HLB) / सिट्रस ग्रीनिंग (एचएलबी) / संत्र्यावरील ग्रीनिंग (एचएलबी)",
        'symptoms': "Blotchy, mottled leaves (asymmetrical yellowing), misshapen and bitter-tasting fruit, and premature fruit drop. Leads to the rapid decline and death of the tree.",
        'cause': "Caused by the bacterium *Candidatus Liberibacter asiaticus*, spread by the Asian Citrus Psyllid insect.",
        'prevention': ["Strict quarantine on citrus plant materials.", "Plant only certified disease-free trees from reputable nurseries.", "Intensive monitoring and control of the Asian Citrus Psyllid insect is paramount."],
        'organic_management': ["Controlling the psyllid vector is key. Regular sprays of neem oil or horticultural oils can deter the insect.", "Release of natural predators of the psyllid, like ladybugs."],
        'chemical_management': ["There is no cure for the disease.", "Management focuses on controlling the psyllid vector with systemic insecticides like Imidacloprid or Thiamethoxam, applied as soil drench or foliar spray.", "**Disclaimer**: Infected trees must be removed. Chemical use is for preventing spread, not curing. Follow local agricultural university guidelines."],
        'source': "Based on recommendations from Central Citrus Research Institute (ICAR-CCRI), Nagpur."
    },

    # == Peach Disease ==
    'Peach___Bacterial_spot': {
        'common_name': "Peach Bacterial Spot / आड़ू का बैक्टीरियल स्पॉट / पीचवरील बॅक्टेरियल स्पॉट (जिवाणूजन्य ठिपके)",
        'symptoms': "Dark, angular spots on leaves that may fall out, creating a 'shot-hole' appearance. Fruit develops pitted, cracked spots.",
        'prevention': ["Plant resistant peach varieties.", "Avoid overhead irrigation.", "Maintain tree vigor with proper nutrients, as stressed trees are more susceptible."],
        'organic_management': ["Apply fixed copper sprays during the dormant season.", "Avoid working around trees when foliage is wet."],
        'chemical_management': ["Dormant season sprays of copper compounds are crucial.", "Sprays containing oxytetracycline (an antibiotic) may be used after petal fall, but effectiveness can be limited. Streptocycline is also used in India.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR & State Horticultural Departments."
    },

    # == Pepper (Bell) Disease ==
    'Pepper,_bell___Bacterial_spot': {
        'common_name': "Bell Pepper Bacterial Spot / शिमला मिर्च का बैक्टीरियल स्पॉट / ढोबळी मिरचीवरील बॅक्टेरियल स्पॉट",
        'symptoms': "Small, water-soaked spots on leaves that turn brown or black with a yellow halo. Spots on fruit are raised and scabby. Can cause significant leaf drop.",
        'prevention': ["Use disease-free seeds and transplants.", "Rotate crops, avoiding planting peppers or tomatoes in the same spot for at least one year.", "Avoid working in the fields when plants are wet."],
        'organic_management': ["Seed treatment with hot water (50°C for 25 minutes) can help.", "Preventative sprays of copper-based bactericides.", "Spraying with a solution of *Bacillus subtilis* can be effective."],
        'chemical_management': ["Spray a combination of Copper Oxychloride and Streptocycline at first sign of disease.", "Alternate sprays with different bactericides to prevent resistance.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from Indian Institute of Horticultural Research (ICAR-IIHR)."
    },

    # == Potato Diseases ==
    'Potato___Early_blight': {
        'common_name': "Potato Early Blight / आलू का अगेती झुलसा / बटाट्याचा लवकर येणारा करपा",
        'symptoms': "Small, dark brown to black spots appear on lower, older leaves. These spots enlarge and show a distinct 'bull's-eye' pattern with concentric rings. The surrounding leaf area may turn yellow.",
        'prevention': ["Plant certified, disease-free seed tubers.", "Practice crop rotation with non-host crops like cereals for at least 2-3 years.", "Maintain balanced soil nutrition. Avoid excess nitrogen.", "Remove and burn crop debris after harvest."],
        'organic_management': ["Seed Treatment: Treat seed tubers with *Trichoderma viride* or *Pseudomonas fluorescens* before planting.", "Neem Cake: Apply neem cake to the soil during field preparation.", "Foliar Spray: Spray a solution of Copper Oxychloride @ 3 grams/liter of water."],
        'chemical_management': ["First Spray: At the sign of disease, spray Mancozeb 75% WP @ 2.5 grams/liter of water.", "Subsequent Sprays: Alternate with Chlorothalonil or Propineb based fungicides every 10-15 days, especially during humid weather.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from Central Potato Research Institute (ICAR-CPRI) & KVKs."
    },
    'Potato___Late_blight': {
        'common_name': "Potato Late Blight / आलू का पछेती झुलसा / बटाट्याचा उशिरा येणारा करपा",
        'symptoms': "Large, dark, water-soaked spots on leaves and stems, often with a white, fuzzy mold on the underside of leaves in humid conditions. It can destroy a crop rapidly. Tubers develop a reddish-brown, dry rot.",
        'prevention': ["Use certified disease-free seed.", "Destroy volunteer potato plants and cull piles.", "Allow for good air circulation and avoid overhead irrigation."],
        'organic_management': ["Preventative sprays of Bordeaux mixture are a traditional and effective method.", "Ensure good drainage to reduce soil moisture.", "Fortify soil with silica-rich amendments like diatomaceous earth."],
        'chemical_management': ["A strict, preventative fungicide spray schedule is critical.", "Prophylactic spray with Mancozeb @ 2.5 g/l.", "After onset, use systemic/contact mixtures like Cymoxanil + Mancozeb or Metalaxyl + Mancozeb.", "**Disclaimer**: This disease spreads extremely fast. Prevention is key. Always follow label instructions."],
        'source': "Based on recommendations from Central Potato Research Institute (ICAR-CPRI)."
    },

    # == Squash Disease ==
    'Squash___Powdery_mildew': {
        'common_name': "Squash Powdery Mildew / स्क्वैश का पाउडरी मिल्ड्यू / भोपळ्यावरील भुरी रोग",
        'symptoms': "White, powdery spots on leaves, stems, and petioles. The spots can spread to cover the entire leaf surface, causing it to turn yellow and die.",
        'prevention': ["Plant resistant varieties.", "Ensure good air circulation and sun exposure by giving plants enough space.", "Avoid overhead watering."],
        'organic_management': ["Spray a solution of potassium bicarbonate (10g per liter water) with a sticker.", "A weekly spray of milk solution (1 part milk to 9 parts water) can be effective.", "Neem oil or horticultural oil sprays also work well."],
        'chemical_management': ["Apply wettable sulfur at the first sign of disease.", "Systemic fungicides like Myclobutanil or Azoxystrobin can be used for severe infections.", "**Disclaimer**: Do not apply sulfur in high temperatures (above 30°C). Always follow label instructions."],
        'source': "Based on general recommendations from ICAR-IARI and KVKs."
    },

    # == Strawberry Disease ==
    'Strawberry___Leaf_scorch': {
        'common_name': "Strawberry Leaf Scorch / स्ट्रॉबेरी का पत्ती झुलसा / स्ट्रॉबेरीवरील पानावरील करपा",
        'symptoms': "Small, purplish, irregular-shaped spots on leaves. These spots enlarge and merge, and the tissue between the veins turns brown, making the leaf look 'scorched'.",
        'prevention': ["Plant resistant varieties.", "Renovate strawberry beds after harvest by mowing off old leaves and removing them.", "Maintain good air circulation with proper plant spacing."],
        'organic_management': ["Remove and destroy infected leaves as soon as they appear.", "Use straw mulch to reduce water splash.", "Copper-based sprays can be used preventatively."],
        'chemical_management': ["Apply fungicides like Captan or Myclobutanil starting early in the spring before the disease becomes established.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR & State Horticultural Departments."
    },

    # == Tomato Diseases ==
    'Tomato___Bacterial_spot': {
        'common_name': "Tomato Bacterial Spot / टमाटर का बैक्टीरियल स्पॉट / टोमॅटोवरील बॅक्टेरियल स्पॉट (जिवाणूजन्य ठिपके)",
        'symptoms': "Small, water-soaked, angular spots on leaves that turn greasy and then brown/black, often with a yellow halo. Fruit lesions are raised and scabby.",
        'prevention': ["Use clean, certified seed.", "Rotate with non-related crops for at least one year.", "Avoid overhead watering and handling wet plants."],
        'organic_management': ["Seed treatment with hot water (50°C for 25 minutes).", "Preventative sprays of copper-based bactericides.", "Spraying with a solution of *Bacillus subtilis* can be effective."],
        'chemical_management': ["Spray a combination of Copper Oxychloride (2.5g/l) and Streptocycline (0.1g/l) at first sign of disease.", "**Disclaimer**: Streptocycline use is regulated in some areas. Always follow local guidelines and label instructions."],
        'source': "Based on recommendations from Indian Institute of Horticultural Research (ICAR-IIHR)."
    },
    'Tomato___Early_blight': {
        'common_name': "Tomato Early Blight / टमाटर का अगेती झुलसा / टोमॅटोचा लवकर येणारा करपा",
        'symptoms': "Dark spots with concentric rings ('bull's-eye' pattern) on lower leaves. A 'collar rot' can form on stems near the soil line. Can cause fruit to rot at the stem end.",
        'prevention': ["Ensure good air circulation with staking and pruning.", "Use mulch to prevent fungal spores from splashing from the soil onto leaves.", "Practice crop rotation."],
        'organic_management': ["Prune off infected lower leaves and destroy them.", "Spray with Bordeaux mixture or Copper Oxychloride.", "Neem cake application in soil is beneficial."],
        'chemical_management': ["Apply fungicides like Mancozeb or Chlorothalonil preventatively.", "In case of attack, use Azoxystrobin or Tebuconazole + Trifloxystrobin.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR-IARI and KVKs."
    },
    'Tomato___Late_blight': {
        'common_name': "Tomato Late Blight / टमाटर का पछेती झुलसा / टोमॅटोचा करपा",
        'symptoms': "Starts as pale green, water-soaked spots on leaves, quickly turning into large, brown-black lesions. A white fuzzy mold can be seen on the underside of leaves in high humidity. Stems get dark lesions, and fruits develop a firm, brown rot. The disease spreads very rapidly.",
        'prevention': ["Use certified, disease-free seeds and seedlings.", "Ensure proper spacing (at least 2 feet) between plants for good air circulation.", "Stake plants to keep leaves and stems off the ground.", "Water the soil at the base of the plant, not the leaves."],
        'organic_management': ["Preventative Spray: Mix 5-10 ml of Neem Oil (300-1500 ppm) with a sticker-spreader in 1 liter of water and spray weekly.", "Application of *Trichoderma viride* or *Pseudomonas fluorescens* to soil.", "Spray sour buttermilk solution (1 liter in 10 liters water)."],
        'chemical_management': ["Preventative Spray: Mancozeb 75% WP @ 2.5 grams/liter of water.", "After Onset: If the disease appears, spray Metalaxyl + Mancozeb combination products @ 2 grams/liter of water.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label. Rotate fungicides to prevent resistance."],
        'source': "Based on recommendations from ICAR & State Agricultural Universities of India."
    },
    'Tomato___Leaf_Mold': {
        'common_name': "Tomato Leaf Mold / टमाटर का लीफ मोल्ड / टोमॅटोवरील पानांची बुरशी",
        'symptoms': "Pale greenish-yellow spots on the upper leaf surface, with velvety, olive-green to brownish spore masses on the underside of the leaf directly below the spots.",
        'prevention': ["Critical to ensure good ventilation and low humidity, especially in polyhouses/greenhouses.", "Stake and prune plants to improve air flow.", "Water at the base of the plant in the morning."],
        'organic_management': ["Regular sprays of neem oil can be effective.", "Spraying with a solution of *Bacillus subtilis* or Copper Oxychloride."],
        'chemical_management': ["Apply fungicides like Chlorothalonil or Mancozeb.", "For severe cases, Azoxystrobin can be used.", "**Disclaimer**: Good ventilation is more effective than any chemical. Always follow label instructions."],
        'source': "Based on general recommendations from ICAR and KVKs."
    },
    'Tomato___Septoria_leaf_spot': {
        'common_name': "Tomato Septoria Leaf Spot / टमाटर का सेप्टोरिया लीफ स्पॉट / टोमॅटोवरील सेप्टोरिया पानावरील ठिपके",
        'symptoms': "Many small, circular spots on lower leaves, with dark brown borders and lighter gray or tan centers. Small black dots (fruiting bodies) can be seen in the center of the spots. Causes heavy leaf drop.",
        'prevention': ["Crop rotation is key (3 years).", "Use mulch to prevent spore splash.", "Control weeds and remove volunteer plants.", "Do not handle wet plants."],
        'organic_management': ["Remove and destroy infected leaves as they appear.", "Apply copper-based fungicides or bio-fungicides containing *Bacillus subtilis*."],
        'chemical_management': ["Apply Mancozeb or Chlorothalonil fungicides as a preventative measure.", "Azoxystrobin can also be effective.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR-IARI and KVKs."
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'common_name': "Tomato Spider Mites / टमाटर पर मकड़ी के कण / टोमॅटोवरील कोळी",
        'symptoms': "Fine, white or yellow stippling (dots) on leaves. As infestation grows, leaves may turn yellow or bronze and fine webbing may be visible on the underside of leaves and between stems. This is a pest, not a disease.",
        'prevention': ["Keep plants well-watered to avoid drought stress.", "Regularly inspect the underside of leaves.", "Encourage natural predators like ladybugs."],
        'organic_management': ["A strong spray of water from a hose can dislodge them.", "Spray with insecticidal soaps or horticultural oils, ensuring good coverage on the underside of leaves.", "Neem oil is also an effective repellent and growth disruptor."],
        'chemical_management': ["Apply specific miticides like Spiromesifen, Propargite, or Dicofol.", "Wettable sulfur can also control mites but should not be used in high heat.", "**Disclaimer**: These are pests, not fungi. Fungicides will not work. Always follow label instructions."],
        'source': "Based on general entomology recommendations from ICAR."
    },
    'Tomato___Target_Spot': {
        'common_name': "Tomato Target Spot / टमाटर का टारगेट स्पॉट / टोमॅटोवरील टार्गेट स्पॉट",
        'symptoms': "Small, water-soaked spots on upper leaves that enlarge into lesions with light brown centers and dark borders, similar to Early Blight but often with a more distinct 'target' look. Can also infect stems and fruit.",
        'prevention': ["Promote good air circulation through staking and pruning.", "Use mulch to prevent spore splash.", "Practice good field sanitation and crop rotation."],
        'organic_management': ["Remove infected leaves promptly.", "Preventative sprays with copper-based fungicides can be helpful."],
        'chemical_management': ["Fungicide applications are the main control method.", "Mancozeb is effective as a preventative.", "Azoxystrobin or other strobilurin fungicides can be used for control.", "**Disclaimer**: Always read and follow the manufacturer's instructions on the product label."],
        'source': "Based on recommendations from ICAR and State Agricultural Universities."
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'common_name': "Tomato Yellow Leaf Curl Virus / टमाटर का पीला पत्ती कर्ल वायरस / टोमॅटोवरील येलो लीफ कर्ल व्हायरस (पिवळा पानावरील गुंडाळी)",
        'symptoms': "Severe stunting of the plant. Upward curling, yellowing, and cupping of leaves. New leaves are much smaller than normal. Drastically reduces fruit production.",
        'cause': "Caused by the *Tomato yellow leaf curl virus* (TYLCV), transmitted by whiteflies.",
        'prevention': ["The most critical step is controlling whitefly populations.", "Use yellow sticky traps to monitor and trap whiteflies.", "Install insect-proof nets in nurseries.", "Remove and destroy infected plants immediately."],
        'organic_management': ["Regular sprays of neem oil or horticultural soap can deter whiteflies.", "Planting marigold as a trap crop around the tomato field can help."],
        'chemical_management': ["There is no cure for the virus.", "Control the whitefly vector. Use systemic insecticides like Imidacloprid for seedlings (soil drench) and spray with Acetamiprid or Diafenthiuron during growth.", "**Disclaimer**: Management focuses entirely on preventing infection by controlling the whitefly. Always follow label instructions."],
        'source': "Based on recommendations from Indian Institute of Horticultural Research (ICAR-IIHR)."
    },
    'Tomato___Tomato_mosaic_virus': {
        'common_name': "Tomato Mosaic Virus / टमाटर का मोज़ेक वायरस / टोमॅटोवरील मोझॅक व्हायरस",
        'symptoms': "Mottled light green and dark green patterns on the leaves (mosaic). Leaves may be malformed, curled, and stunted. Overall plant growth is often stunted.",
        'cause': "Caused by the *Tomato mosaic virus* (ToMV). Highly stable and easily transmitted by touch.",
        'prevention': ["Wash hands thoroughly with soap before handling plants.", "Do not smoke or use tobacco products around tomato plants, as the virus is related to Tobacco Mosaic Virus.", "Control weeds and remove infected plants.", "Disinfect tools regularly."],
        'organic_management': ["There is no cure. Prevention through sanitation is the only organic approach.", "Soaking seeds in a 10% solution of trisodium phosphate (TSP) for 15 minutes can inactivate the virus on the seed coat."],
        'chemical_management': ["There is no chemical cure for viral diseases. Remove and destroy infected plants to prevent spread.", "**Disclaimer**: Focus on prevention and sanitation."],
        'source': "Based on general plant virology recommendations from ICAR."
    }
}


# --- 2. Translation Database (Hindi & Marathi) ---
# This dictionary stores all necessary translations.
TRANSLATIONS = {
    'hi': {
        # UI Elements
        'app_title': "कृषि-रक्षक: पादप रोग सलाहकार",
        'disease_information': "रोग की जानकारी",
        'common_name': "सामान्य नाम",
        'symptoms': "लक्षण",
        'prevention': "रोकथाम के उपाय",
        'organic_management': "जैविक उपचार",
        'chemical_management': "रासायनिक उपचार",
        'source': "स्रोत",
        'disclaimer': "**अस्वीकरण**: हमेशा उत्पाद लेबल पर दिए गए निर्माता के निर्देशों को पढ़ें और उनका पालन करें।",
        'plant_is_healthy': "पौधा स्वस्थ है। नियमित निगरानी और देखभाल जारी रखें।",
        # Disease Content (Add translations for all 26 diseases here)
        'Apple___Apple_scab': {'symptoms': "पत्तियों पर मखमली, जैतून-हरे धब्बे, जो बाद में काले हो जाते हैं और पत्ती को मोड़ देते हैं। फल पर, यह गहरे, कॉर्क जैसे, खुरदुरे धब्बे बनाता है, जिससे फल विकृत और बेचने लायक नहीं रहता।"},
        'Apple___Black_rot': {'symptoms': "'मेंढक-आंख' पत्ती धब्बे (बैंगनी धब्बे जो भूरे रंग के घेरे में बड़े हो जाते हैं), शाखाओं पर कैंकर, और फल पर एक कठोर, भूरे-से-काले रंग की सड़न, जो अक्सर फूल के सिरे से शुरू होती है।"},
        'Apple___Cedar_apple_rust': {'symptoms': "पत्तियों और फलों पर चमकीले पीले-नारंगी धब्बे। पत्तियों के नीचे की तरफ, छोटी कप जैसी संरचनाएं बन सकती हैं। इसे अपने जीवन चक्र को पूरा करने के लिए पास में एक जुनिपर या देवदार के पेड़ की आवश्यकता होती है।"},
        'Cherry_(including_sour)___Powdery_mildew': {'symptoms': "पत्तियों और टहनियों पर सफेद, पाउडर जैसे धब्बे। संक्रमित पत्तियां विकृत या मुड़ सकती हैं। फल को भी प्रभावित कर सकता है, जिससे उस पर खुरदरापन आ जाता है।"},
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'symptoms': "छोटे, पानी से भरे धब्बों के रूप में शुरू होता है जो लंबे, संकीर्ण, आयताकार भूरे या ग्रे घावों में विकसित होते हैं, जो आमतौर पर पत्ती की नसों के समानांतर होते हैं।"},
        'Corn_(maize)___Common_rust_': {'symptoms': "पत्ती की ऊपरी और निचली दोनों सतहों पर छोटे, दालचीनी-भूरे, पाउडर जैसे फफोले। फफोले अंडाकार या लम्बे होते हैं।"},
        'Corn_(maize)___Northern_Leaf_Blight': {'symptoms': "पत्तियों पर बड़े, सिगार के आकार के, भूरे-हरे से लेकर पीले रंग के घाव, जो आमतौर पर 1 से 6 इंच लंबे होते हैं।"},
        'Grape___Black_rot': {'symptoms': "पत्तियों पर छोटे, भूरे रंग के गोलाकार धब्बे जिनके चारों ओर काली सीमा विकसित होती है। फल पर, यह एक हल्के धब्बे के रूप में शुरू होता है, फिर पूरा बेर सड़ जाता है, काला हो जाता है, और एक कठोर 'ममी' में सिकुड़ जाता है।"},
        'Grape___Esca_(Black_Measles)': {'symptoms': "यह पुराना (बेर पर छोटे, गहरे धब्बे, पत्तियों पर 'बाघ-धारी' पैटर्न) या तीव्र (बेल का अचानक मुरझाना और मर जाना) हो सकता है।"},
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'symptoms': "पत्तियों पर अनियमित, गहरे लाल-भूरे से काले धब्बे, जिसके कारण पत्तियां समय से पहले गिर सकती हैं। धब्बों के नीचे की तरफ कालिख जैसा दिख सकता है।"},
        'Orange___Haunglongbing_(Citrus_greening)': {'symptoms': "धब्बेदार, चित्तीदार पत्तियां (असममित पीलापन), विकृत और कड़वे स्वाद वाले फल, और समय से पहले फल गिरना। यह पेड़ के तेजी से पतन और मृत्यु का कारण बनता है।"},
        'Peach___Bacterial_spot': {'symptoms': "पत्तियों पर गहरे, कोणीय धब्बे जो गिर सकते हैं, जिससे 'शॉट-होल' जैसा दिखता है। फल पर गड्ढेदार, फटे हुए धब्बे विकसित होते हैं।"},
        'Pepper,_bell___Bacterial_spot': {'symptoms': "पत्तियों पर छोटे, पानी से भरे धब्बे जो पीले प्रभामंडल के साथ भूरे या काले हो जाते हैं। फल पर धब्बे उभरे हुए और खुरदरे होते हैं। इससे पत्तियों का भारी मात्रा में गिरना हो सकता है।"},
        'Potato___Early_blight': {'symptoms': "निचली, पुरानी पत्तियों पर छोटे, गहरे भूरे से काले धब्बे दिखाई देते हैं। ये धब्बे बड़े हो जाते हैं और संकेंद्रित छल्लों का एक विशिष्ट 'बैल की आंख' पैटर्न दिखाते हैं।"},
        'Potato___Late_blight': {'symptoms': "पत्तियों और तनों पर बड़े, गहरे, पानी से भरे धब्बे, अक्सर उच्च आर्द्रता में पत्तियों के नीचे की तरफ एक सफेद, रोएंदार फफूंदी के साथ। यह फसल को तेजी से नष्ट कर सकता है।"},
        'Squash___Powdery_mildew': {'symptoms': "पत्तियों, तनों और डंठलों पर सफेद, पाउडर जैसे धब्बे। ये धब्बे पूरी पत्ती की सतह को ढकने के लिए फैल सकते हैं, जिससे वह पीली होकर मर जाती है।"},
        'Strawberry___Leaf_scorch': {'symptoms': "पत्तियों पर छोटे, बैंगनी, अनियमित आकार के धब्बे। ये धब्बे बड़े होकर विलीन हो जाते हैं, और नसों के बीच का ऊतक भूरा हो जाता है, जिससे पत्ती 'झुलसी हुई' दिखती है।"},
        'Tomato___Bacterial_spot': {'symptoms': "पत्तियों पर छोटे, पानी से भरे, कोणीय धब्बे जो चिकने हो जाते हैं और फिर भूरे/काले हो जाते हैं, अक्सर एक पीले प्रभामंडल के साथ। फल पर घाव उभरे हुए और खुरदरे होते हैं।"},
        'Tomato___Early_blight': {'symptoms': "निचली पत्तियों पर संकेंद्रित छल्लों ('बैल की आंख' पैटर्न) के साथ गहरे धब्बे। मिट्टी के पास तनों पर 'कॉलर रॉट' बन सकता है।"},
        'Tomato___Late_blight': {'symptoms': "पत्तियों पर हल्के हरे, पानी से भरे धब्बे के रूप में शुरू होता है, जो जल्दी से बड़े, भूरे-काले घावों में बदल जाते हैं। अधिक नमी में पत्तियों के नीचे की तरफ एक सफेद फफूंदी देखी जा सकती है।"},
        'Tomato___Leaf_Mold': {'symptoms': "पत्ती की ऊपरी सतह पर हल्के हरे-पीले धब्बे, पत्ती के नीचे सीधे धब्बों के नीचे मखमली, जैतून-हरे से भूरे रंग के बीजाणु के साथ।"},
        'Tomato___Septoria_leaf_spot': {'symptoms': "निचली पत्तियों पर कई छोटे, गोलाकार धब्बे, जिनके गहरे भूरे रंग के किनारे और हल्के ग्रे या पीले केंद्र होते हैं। धब्बों के केंद्र में छोटे काले बिंदु देखे जा सकते हैं।"},
        'Tomato___Spider_mites Two-spotted_spider_mite': {'symptoms': "पत्तियों पर महीन, सफेद या पीले रंग के धब्बे। जैसे-जैसे संक्रमण बढ़ता है, पत्तियां पीली या कांस्य रंग की हो सकती हैं और पत्तियों के नीचे और तनों के बीच महीन जाला दिखाई दे सकता है।"},
        'Tomato___Target_Spot': {'symptoms': "ऊपरी पत्तियों पर छोटे, पानी से भरे धब्बे जो हल्के भूरे केंद्रों और गहरे किनारों वाले घावों में बड़े हो जाते हैं, जो अगेती झुलसा के समान होते हैं लेकिन अक्सर अधिक विशिष्ट 'लक्ष्य' रूप के साथ।"},
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'symptoms': "पौधे का गंभीर बौनापन। पत्तियों का ऊपर की ओर मुड़ना, पीला पड़ना और कप का आकार लेना। नई पत्तियां सामान्य से बहुत छोटी होती हैं।"},
        'Tomato___Tomato_mosaic_virus': {'symptoms': "पत्तियों पर हल्के हरे और गहरे हरे रंग के पैटर्न (मोज़ेक)। पत्तियां विकृत, मुड़ी हुई और बौनी हो सकती हैं।"}
    },
    'mr': {
        # UI Elements
        'app_title': "कृषी-रक्षक: वनस्पती रोग सल्लागार",
        'disease_information': "रोगाची माहिती",
        'common_name': "सामान्य नाव",
        'symptoms': "लक्षणे",
        'prevention': "प्रतिबंधात्मक उपाय",
        'organic_management': "सेंद्रिय व्यवस्थापन",
        'chemical_management': "रासायनिक व्यवस्थापन",
        'source': "स्रोत",
        'disclaimer': "**अस्वीकरण**: डोस आणि सुरक्षिततेच्या खबरदारीसाठी नेहमी उत्पादनाच्या लेबलवरील निर्मात्याच्या सूचना वाचा आणि त्यांचे पालन करा.",
        'plant_is_healthy': "रोप निरोगी आहे. नियमित निरीक्षण आणि काळजी घेणे सुरू ठेवा.",
        # Disease Content (Add translations for all 26 diseases here)
        'Apple___Apple_scab': {'symptoms': "पानांवर मखमली, जैतुनी-हिरवे ठिपके, जे नंतर काळे होतात आणि पान कुरळे होते. फळांवर, ते गडद, कॉर्कसारखे, खरबरीत डाग तयार करते, ज्यामुळे फळ विकृत आणि विक्रीसाठी अयोग्य होते."},
        'Apple___Black_rot': {'symptoms': "'बेडकाच्या डोळ्यासारखे' पानांवरील ठिपके (जांभळे ठिपके जे तपकिरी वर्तुळात मोठे होतात), फांद्यांवर कॅन्कर आणि फळांवर एक घट्ट, तपकिरी-ते-काळी सड, जी अनेकदा फुलाच्या टोकापासून सुरू होते."},
        'Apple___Cedar_apple_rust': {'symptoms': "पानांवर आणि फळांवर चमकदार पिवळे-नारंगी ठिपके. पानांच्या खालच्या बाजूला, लहान कपसारख्या रचना तयार होऊ शकतात. याला त्याचे जीवनचक्र पूर्ण करण्यासाठी जवळच्या जुनिपर किंवा देवदार वृक्षाची आवश्यकता असते."},
        'Cherry_(including_sour)___Powdery_mildew': {'symptoms': "पानांवर आणि फांद्यांवर पांढरे, भुकटीसारखे डाग. संक्रमित पाने विकृत किंवा कुरळी होऊ शकतात. फळांवरही परिणाम होऊ शकतो, ज्यामुळे त्यावर खरबरीतपणा येतो."},
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'symptoms': "लहान, पाण्याने भिजलेल्या ठिपक्यांपासून सुरुवात होते जे लांब, अरुंद, आयताकृती तपकिरी किंवा राखाडी रंगाच्या जखमांमध्ये विकसित होतात, सामान्यतः पानाच्या शिरांच्या समांतर."},
        'Corn_(maize)___Common_rust_': {'symptoms': "पानाच्या वरच्या आणि खालच्या दोन्ही पृष्ठभागांवर लहान, दालचिनी-तपकिरी, भुकटीसारखे फोड. फोड अंडाकृती किंवा लांबट असतात."},
        'Corn_(maize)___Northern_Leaf_Blight': {'symptoms': "पानांवर मोठे, सिगारच्या आकाराचे, राखाडी-हिरवे ते पिवळसर रंगाचे घाव, जे सामान्यतः १ ते ६ इंच लांब असतात."},
        'Grape___Black_rot': {'symptoms': "पानांवर लहान, तपकिरी गोलाकार ठिपके ज्यांच्याभोवती काळी किनार तयार होते. फळावर, ते फिकट डागाच्या रूपात सुरू होते, नंतर संपूर्ण मणी सडतो, काळा होतो आणि कडक 'ममी'मध्ये आकसतो."},
        'Grape___Esca_(Black_Measles)': {'symptoms': "हे दीर्घकालीन (मण्यांवर लहान, गडद ठिपके, पानांवर 'वाघाच्या पट्ट्यांसारखे' नमुने) किंवा तीव्र (वेलीचे अचानक कोमेजणे आणि मरणे) असू शकते."},
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'symptoms': "पानांवर अनियमित, गडद लालसर-तपकिरी ते काळे डाग, ज्यामुळे पाने अकाली गळून पडू शकतात. डागांच्या खालच्या बाजूला काजळीसारखे दिसू शकते."},
        'Orange___Haunglongbing_(Citrus_greening)': {'symptoms': "ठिपकेदार, चित कबरी पाने (असममित पिवळेपणा), विकृत आणि कडू चवीची फळे, आणि अकाली फळगळ. यामुळे झाडाचा झपाट्याने ऱ्हास होतो आणि मृत्यू होतो."},
        'Peach___Bacterial_spot': {'symptoms': "पानांवर गडद, ​​कोनीय ठिपके जे गळून पडू शकतात, ज्यामुळे 'शॉट-होल' सारखे स्वरूप येते. फळांवर खड्डेयुक्त, तडकलेले डाग विकसित होतात."},
        'Pepper,_bell___Bacterial_spot': {'symptoms': "पानांवर लहान, पाण्याने भिजलेले डाग जे पिवळ्या प्रभामंडलासह तपकिरी किंवा काळे होतात. फळांवरील डाग उंच आणि खरबरीत असतात. यामुळे मोठ्या प्रमाणात पानगळ होऊ शकते."},
        'Potato___Early_blight': {'symptoms': "खालच्या, जुन्या पानांवर लहान, गडद तपकिरी ते काळे ठिपके दिसतात. हे ठिपके मोठे होतात आणि त्यामध्ये एकाग्र वर्तुळांचा एक विशिष्ट 'बैलाच्या डोळ्यासारखा' नमुना दिसतो."},
        'Potato___Late_blight': {'symptoms': "पानांवर आणि देठांवर मोठे, गडद, पाण्याने भिजलेले डाग, अनेकदा उच्च आर्द्रतेमध्ये पानांच्या खालच्या बाजूला पांढऱ्या, केसाळ बुरशीसह. हे पीक वेगाने नष्ट करू शकते."},
        'Squash___Powdery_mildew': {'symptoms': "पाने, देठ आणि देठांवर पांढरे, भुकटीसारखे डाग. हे डाग संपूर्ण पानाच्या पृष्ठभागावर पसरू शकतात, ज्यामुळे ते पिवळे पडून मरते."},
        'Strawberry___Leaf_scorch': {'symptoms': "पानांवर लहान, जांभळ्या रंगाचे, अनियमित आकाराचे डाग. हे डाग मोठे होऊन एकत्र येतात आणि शिरांमधील ऊतक तपकिरी होते, ज्यामुळे पान 'भाजल्यासारखे' दिसते."},
        'Tomato___Bacterial_spot': {'symptoms': "पानांवर लहान, पाण्याने भिजलेले, कोनीय डाग जे तेलकट होतात आणि नंतर तपकिरी/काळे होतात, अनेकदा पिवळ्या प्रभामंडलासह. फळांवरील जखमा उंच आणि खरबरीत असतात."},
        'Tomato___Early_blight': {'symptoms': "खालच्या पानांवर एकाग्र वर्तुळांसह ('बैलाच्या डोळ्यासारखा' नमुना) गडद डाग. जमिनीजवळील देठांवर 'कॉलर रॉट' तयार होऊ शकतो."},
        'Tomato___Late_blight': {'symptoms': "सुरुवातीला पानांवर फिकट हिरवे, पाण्याने भिजल्यासारखे ठिपके दिसतात, जे लवकरच मोठ्या, तपकिरी-काळ्या रंगाच्या डागांमध्ये बदलतात. जास्त आर्द्रतेमध्ये पानांच्या खालील बाजूस पांढरी बुरशी दिसू शकते."},
        'Tomato___Leaf_Mold': {'symptoms': "पानाच्या वरच्या पृष्ठभागावर फिकट हिरवे-पिवळे डाग, पानाच्या खाली थेट डागांच्या खाली मखमली, जैतुनी-हिरव्या ते तपकिरी रंगाच्या बुरशीच्या पुंजक्यांसह."},
        'Tomato___Septoria_leaf_spot': {'symptoms': "खालच्या पानांवर अनेक लहान, गोलाकार डाग, ज्यांच्या गडद तपकिरी कडा आणि हलक्या राखाडी किंवा पिवळसर केंद्र असतात. डागांच्या मध्यभागी लहान काळे ठिपके दिसू शकतात."},
        'Tomato___Spider_mites Two-spotted_spider_mite': {'symptoms': "पानांवर बारीक, पांढरे किंवा पिवळे ठिपके. प्रादुर्भाव वाढल्यास, पाने पिवळी किंवा कांस्य रंगाची होऊ शकतात आणि पानांच्या खाली आणि देठांच्या मध्ये बारीक जाळे दिसू शकते."},
        'Tomato___Target_Spot': {'symptoms': "वरच्या पानांवर लहान, पाण्याने भिजलेले डाग जे हलक्या तपकिरी केंद्र आणि गडद कडा असलेल्या जखमांमध्ये मोठे होतात, जे लवकर येणाऱ्या करप्यासारखे असतात परंतु अनेकदा अधिक विशिष्ट 'लक्ष्य' स्वरूपासह."},
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'symptoms': "रोपाची तीव्र वाढ खुंटणे. पानांचे वरच्या दिशेने कुरळे होणे, पिवळे पडणे आणि वाटीसारखे होणे. नवीन पाने सामान्यपेक्षा खूपच लहान असतात."},
        'Tomato___Tomato_mosaic_virus': {'symptoms': "पानांवर फिकट हिरव्या आणि गडद हिरव्या रंगाचे मिश्रण (मोझॅक). पाने विकृत, कुरळी आणि खुंटलेली असू शकतात."}
    }
}


# --- 3. Main Advisory Script Logic ---

class AdvisoryService:
    def __init__(self, advisory_data, translation_data):
        self.advisories = advisory_data
        self.translations = translation_data

    def _get_translation(self, key: str, lang: str, default_text: str = "") -> str:
        """Helper to get a translated string, falling back to the key name."""
        return self.translations.get(lang, {}).get(key, default_text or key.replace('_', ' ').title())

    def generate_advisory(self, disease_key: str, lang: str = 'en') -> str:
        """
        Generates a formatted, translated advisory string for a given disease.
        """
        # --- Handle Healthy Case ---
        plant_name = disease_key.split('___')[0].replace('_', ' ')
        if 'healthy' in disease_key:
            healthy_message = self._get_translation('plant_is_healthy', lang)
            # A more dynamic healthy message
            if lang == 'hi':
                healthy_message = f"{plant_name} का पौधा स्वस्थ है। नियमित निगरानी और देखभाल जारी रखें।"
            elif lang == 'mr':
                healthy_message = f"{plant_name} चे रोप निरोगी आहे. नियमित निरीक्षण आणि काळजी घेणे सुरू ठेवा."
            else:
                healthy_message = f"The {plant_name} plant is healthy. Continue regular monitoring and care."
            return f"--- {self._get_translation('app_title', lang)} ---\n\n✅ {healthy_message}\n"

        # --- Get Data and Translations for the Diseased Case ---
        advisory_en = self.advisories.get(disease_key)
        if not advisory_en:
            return "Advisory not available for this disease."

        # Get the translated dictionary for the specific disease, fall back to English for symptoms if not found
        advisory_translated = self.translations.get(lang, {}).get(disease_key, {})

        def format_list(items):
            return "\n".join([f"  • {item}" for item in items])

        # --- Build the Output String ---
        output = [
            f"--- {self._get_translation('app_title', lang)} ---",
            f"\n\n📋 **{self._get_translation('disease_information', lang)}**",
            "--------------------",
            f"**{self._get_translation('common_name', lang)}**: {advisory_en.get('common_name', 'N/A')}",
            f"\n**{self._get_translation('symptoms', lang)}**:",
            f"{advisory_translated.get('symptoms', advisory_en.get('symptoms'))}",
            f"\n**{self._get_translation('prevention', lang)}**:",
            format_list(advisory_en.get('prevention', [])),
            f"\n**{self._get_translation('organic_management', lang)}**:",
            format_list(advisory_en.get('organic_management', [])),
            f"\n**{self._get_translation('chemical_management', lang)}**:",
            format_list(advisory_en.get('chemical_management', [])),
            f"\n\n*{self._get_translation('disclaimer', lang)}*",
            f"\n**{self._get_translation('source', lang)}**: {advisory_en.get('source')}"
        ]
        return "\n".join(output)

# --- 4. Demonstration ---

if __name__ == "__main__":
    advisory_service = AdvisoryService(ADVISORY_DATA, TRANSLATIONS)
    
    # Example 1: Tomato Late Blight in English
    print("="*60)
    print("DEMO 1: Tomato Late Blight (English)")
    print("="*60)
    print(advisory_service.generate_advisory('Tomato___Late_blight', lang='en'))
    
    # Example 2: Potato Early Blight in Hindi
    print("\n" + "="*60)
    print("DEMO 2: Potato Early Blight (हिंदी में सलाह)")
    print("="*60)
    print(advisory_service.generate_advisory('Potato___Early_blight', lang='hi'))
    
    # Example 3: Grape Black Rot in Marathi
    print("\n" + "="*60)
    print("DEMO 3: Grape Black Rot (मराठीत सल्ला)")
    print("="*60)
    print(advisory_service.generate_advisory('Grape___Black_rot', lang='mr'))
    
    # Example 4: Healthy Corn plant
    print("\n" + "="*60)
    print("DEMO 4: Healthy Corn Plant (English)")
    print("="*60)
    print(advisory_service.generate_advisory('Corn_(maize)___healthy', lang='en'))