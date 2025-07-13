import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model('dog_breed_identifier/models/dog_breed_model.h5')

# Load class indices mapping
with open('dog_breed_identifier/models/class_indices.json') as f:
    class_indices = json.load(f)
# Reverse the mapping: index -> breed name
idx_to_breed = {v: k for k, v in class_indices.items()}

breed_descriptions = {
    "n02085620-Chihuahua": "The Chihuahua, the smallest dog breed 🐶, hails from Mexico with a bold, lively personality. Tiny but confident, it’s fiercely loyal and bonds tightly with owners. Its smooth or long coat, big eyes, and apple-shaped head make it a charming, devoted companion. 🥰",
    "n02085782-Japanese_spaniel": "The Japanese Chin, a graceful toy breed 🐾, is elegant and cat-like from Asia. Affectionate and charming, it loves human companionship. Its silky black-and-white or red-and-white coat and expressive face make it a royal favorite and delightful pet. 😺",
    "n02085936-Maltese_dog": "The Maltese, a playful toy breed 🐕, boasts a long, silky white coat and gentle nature. From the Mediterranean, it thrives on attention and suits families. Its lively personality and elegant look make it a beloved companion, needing regular grooming. 🛁",
    "n02086079-Pekinese": "The Pekinese, a regal Chinese toy breed 👑, has a lion-like mane and confident spirit. Loyal and affectionate, its luxurious coat comes in various colors. With a flat face and small size, it’s a cherished companion needing regular grooming. 🦁",
    "n02086240-Shih-Tzu": "The Shih Tzu, a Tibetan toy breed 🐶, is friendly and affectionate with a flowing coat. Its chrysanthemum-like face adds charm. Playful yet calm, it thrives in loving homes, requiring regular grooming to maintain its stunning appearance. 💖",
    "n02086646-Blenheim_spaniel": "The Blenheim Spaniel, a cheerful Cavalier 🐾, is affectionate with a silky chestnut-and-white coat. Gentle and great with kids, it loves family life. Its expressive eyes and elegant demeanor make it a loyal, delightful companion for all. 🥰",
    "n02086910-papillon": "The Papillon, a dainty toy breed 🦋, is named for its butterfly-like ears. Lively and intelligent, it excels in agility. Its silky coat has colorful markings. Affectionate and alert, Papillons are ideal for active households. 🏃‍♂️",
    "n02087046-toy_terrier": "The Toy Terrier, a spirited small breed 🐶, is energetic and loyal with a sleek coat. Resembling a mini Doberman, it’s bold and loves attention. Perfect for active owners, this tiny companion thrives on play and devotion. 🎉",
    "n02087394-Rhodesian_ridgeback": "The Rhodesian Ridgeback, a strong African hound 🦒, has a unique ridge of hair. Loyal and protective, it’s a skilled hunter and guardian. Its athletic build and short coat suit active owners needing a devoted, energetic companion. 🏋️‍♂️",
    "n02088094-Afghan_hound": "The Afghan Hound, an elegant sighthound 🐕‍🦺, boasts a flowing coat and regal demeanor. Independent yet loyal, it’s built for speed. From Afghanistan, it needs grooming and exercise, perfect for those admiring its aristocratic charm. ✨",
    "n02088238-basset": "The Basset Hound, a low-slung scent hound 🐾, is gentle with a soulful expression. Its long ears and droopy eyes are iconic. Known for tracking, it’s a calm companion for relaxed homes, loving scent adventures. 🐽",
    "n02088364-beagle": "The Beagle, a friendly scent hound 🐶, is curious and energetic with a keen nose. Its tricolor coat and expressive face charm families. Loving adventure, Beagles need exercise and are cheerful, loyal companions. 🏞️",
    "n02088466-bloodhound": "The Bloodhound, a large scent hound 🕵️‍♂️, excels in tracking with its wrinkled face and droopy ears. Gentle and loyal, its short coat is easy to maintain. Determined and affectionate, it’s ideal for owners valuing its unique skills. 🐾",
    "n02088632-bluetick": "The Bluetick Coonhound, a skilled hunter 🦝, is loyal with a striking blue-ticked coat. Its deep bark aids tracking raccoons. Friendly and energetic, Blueticks need active owners and space to thrive as devoted companions. 🎶",
    "n02089078-black-and-tan_coonhound": "The Black-and-Tan Coonhound, a tenacious hunter 🦌, is friendly with a sleek coat. Its deep bark and tracking skills shine in hunting. Energetic and loyal, it suits active owners seeking a vocal, devoted companion. 🐕",
    "n02089867-Walker_hound": "The Treeing Walker Coonhound, a swift hound 🦴, excels in tracking with a tricolor coat. Friendly and energetic, it loves hunting. Loyal and sociable, it needs exercise and space, thriving as a devoted companion. 🏃‍♂️",
    "n02089973-English_foxhound": "The English Foxhound, a pack hound 🦊, is energetic and friendly, bred for fox hunting. Its tricolor coat and athletic build suit active lifestyles. Loyal and sociable, it needs ample exercise, ideal for outdoor-loving owners. 🌳",
    "n02090379-redbone": "The Redbone Coonhound, a sleek hunter 🦝, is loyal with a striking red coat. Its tracking prowess and deep bark shine in hunting. Friendly and energetic, Redbones need active owners and space as devoted companions. 🔥",
    "n02090622-borzoi": "The Borzoi, a graceful Russian sighthound 🐕‍🦺, is calm with a silky coat. Built for speed, its slender frame exudes elegance. Independent yet loyal, it needs exercise and grooming, ideal for those loving refined beauty. ✨",
    "n02090721-Irish_wolfhound": "The Irish Wolfhound, a gentle giant 🐺, is one of the tallest breeds. Calm and loyal, its wiry coat suits its massive size. Bred for hunting wolves, it thrives in spacious, loving homes with moderate exercise. 🏰",
    "n02091032-Italian_greyhound": "The Italian Greyhound, a delicate sighthound 🐶, is affectionate with a sleek coat. Its slender frame loves to cuddle. Elegant and graceful, it needs minimal grooming and exercise, perfect for warm, loving homes. 🛋️",
    "n02091134-whippet": "The Whippet, a slender sighthound 🏃‍♂️, is gentle and fast with a smooth coat. Calm indoors, it loves running. Affectionate and loyal, Whippets need exercise and warmth, thriving in active households as devoted companions. ⚡",
    "n02091244-Ibizan_hound": "The Ibizan Hound, a Spanish sighthound 🐇, is graceful and alert, excelling in rabbit hunting. Its sleek coat highlights elegance. Independent yet loyal, it needs active owners and exercise to thrive as a devoted companion. 🌞",
    "n02091467-Norwegian_elkhound": "The Norwegian Elkhound, a sturdy spitz 🦌, is loyal and bold, bred for elk hunting. Its silver-gray coat suits cold climates. Energetic and intelligent, it needs exercise and stimulation, thriving in active homes. ❄️",
    "n02091635-otterhound": "The Otterhound, a rare scent hound 🦦, is friendly with a shaggy coat for water. Bred for otter hunting, it loves swimming. Playful and loyal, Otterhounds need active owners and space as unique companions. 🏊‍♂️",
    "n02091831-Saluki": "The Saluki, a Middle Eastern sighthound 🐕‍🦺, is graceful and fast with a silky coat. Built for endurance, it’s independent yet loyal. Requiring exercise and minimal grooming, Salukis suit owners valuing their aristocratic charm. 🏜️",
    "n02092002-Scottish_deerhound": "The Scottish Deerhound, a noble sighthound 🦌, is gentle, bred for deer coursing. Its wiry coat and tall frame are regal. Calm yet athletic, it needs space and exercise, thriving as a loyal companion in active homes. 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "n02092339-Weimaraner": "The Weimaraner, a sleek gundog 🐕, is energetic with a silver-gray coat. Bred for hunting, it’s loyal and intelligent. Needing ample exercise and minimal grooming, Weimaraners thrive as devoted companions for active owners. 🌟",
    "n02093256-Staffordshire_bullterrier": "The Staffordshire Bull Terrier, a muscular breed 💪, is loyal and loves people. Its smooth coat and strong build highlight athleticism. Playful and affectionate, Staffies need exercise and training, thriving in active families. 🥰",
    "n02093428-American_Staffordshire_terrier": "The American Staffordshire Terrier, a strong breed 🦾, is loyal and affectionate. Its muscular build and short coat are imposing. Courageous and loving, AmStaffs need exercise and training, thriving as protective companions in active homes. 🏡",
    "n02093647-Bedlington_terrier": "The Bedlington Terrier, a unique breed 🐑, resembles a lamb with a curly coat. Intelligent and spirited, it was bred for hunting. Loyal and affectionate, Bedlingtons need grooming and exercise, ideal for those loving their elegant look. 🥳",
    "n02093754-Border_terrier": "The Border Terrier, a small working breed 🦴, is plucky with a wiry coat. Bred for fox hunting, it’s energetic and affectionate. Loyal and adaptable, Borders need moderate exercise and grooming, thriving in active, loving homes. 🏞️",
    "n02093859-Kerry_blue_terrier": "The Kerry Blue Terrier, an Irish breed 🐕, is spirited with a soft, blue-gray coat. Loyal and intelligent, it excels in various roles. Kerries need grooming and exercise, thriving as affectionate companions for active owners. 🇮🇪",
    "n02093991-Irish_terrier": "The Irish Terrier, a fiery breed 🔥, is loyal with a wiry red coat. Known as a ‘daredevil,’ it’s courageous and energetic. Affectionate with families, it needs exercise and grooming, thriving as a spirited companion. 🐾",
    "n02094114-Norfolk_terrier": "The Norfolk Terrier, a sturdy breed 🦴, is fearless with a wiry coat and folded ears. Bred for hunting, it loves digging. Loyal and adaptable, Norfolks need moderate exercise and grooming, thriving in active families. 🥰",
    "n02094258-Norwich_terrier": "The Norwich Terrier, a compact breed 🐶, is bold with a wiry coat and pricked ears. Bred for hunting, it’s energetic and affectionate. Norwich Terriers need exercise and grooming, thriving as charming companions in active homes. 🏃‍♂️",
    "n02094433-Yorkshire_terrier": "The Yorkshire Terrier, a tiny toy breed 🐾, is spirited with a silky blue-and-tan coat. Bold and affectionate, it’s a loyal companion. Yorkies need grooming and moderate exercise, thriving in loving homes that cherish their charm. 💖",
    "n02095314-wire-haired_fox_terrier": "The Wire-Haired Fox Terrier, a bold breed 🦊, is energetic with a wiry coat. Intelligent and loves hunting, it’s loyal and spirited. Needing grooming and exercise, it thrives in active homes as a lively companion. 🏞️",
    "n02095570-Lakeland_terrier": "The Lakeland Terrier, a tough breed 🦴, is confident with a wiry coat. Bred for rugged terrain, it’s energetic and loyal. Lakelands need grooming and exercise, thriving as spirited companions for active owners. ⛰️",
    "n02095889-Sealyham_terrier": "The Sealyham Terrier, a rare breed 🐶, is calm with a white, wiry coat. Bred for hunting, it’s loyal and adaptable. Sealyhams need grooming and moderate exercise, thriving as charming companions for attentive owners. 🥰",
    "n02096051-Airedale": "The Airedale Terrier, the largest terrier 🦾, is intelligent and courageous with a wiry coat. Versatile and loyal, it excels in many roles. Airedales need grooming and exercise, thriving as confident companions for active owners. 🌟",
    "n02096177-cairn": "The Cairn Terrier, a hardy breed 🦴, is spirited with a shaggy coat. Bred for chasing prey, it’s energetic and loyal. Cairns need moderate exercise and grooming, thriving as charming companions in active, loving homes. 🏞️",
    "n02096294-Australian_terrier": "The Australian Terrier, a tough breed 🐾, is loyal with a wiry blue-and-tan coat. Bold and energetic, it loves hunting. Affectionate and adaptable, Aussies need exercise and grooming, thriving as devoted companions. 🇦🇺",
    "n02096437-Dandie_Dinmont": "The Dandie Dinmont Terrier, a unique breed 🐶, is friendly with a topknot and long body. Its crisp coat needs grooming. Loyal and calm, Dandies need moderate exercise, thriving as charming companions for attentive owners. 🥳",
    "n02096585-Boston_bull": "The Boston Terrier, a compact breed 🐕, is friendly with a tuxedo-like coat. Intelligent and playful, it’s great with families. Bostons need minimal grooming and exercise, thriving as charming companions in various homes. 🎩",
    "n02097047-miniature_schnauzer": "The Miniature Schnauzer, a small terrier 🐾, is spirited with a wiry coat. Alert and intelligent, it’s a great watchdog. Affectionate and energetic, Minis need grooming and exercise, thriving in active homes. 🔔",
    "n02097130-giant_schnauzer": "The Giant Schnauzer, a powerful breed 🦾, is loyal with a wiry coat. Bred for guarding, it’s intelligent and protective. Giants need grooming and exercise, thriving as devoted companions for active, experienced owners. 🛡",
    "n02097209-standard_schnauzer": "The Standard Schnauzer, a versatile breed 🐶, is bold with a wiry coat. Intelligent and loyal, it’s a great guard dog. Standards need grooming and exercise, thriving as protective companions for active owners. 🛠",
    "n02097298-Scotch_terrier": "The Scotch Terrier, a sturdy breed 🦴, is independent with a wiry coat. Bold and loyal, its distinctive silhouette shines. Scotties need grooming and moderate exercise, thriving as confident companions in loving homes. 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "n02097474-Tibetan_terrier": "The Tibetan Terrier, a shaggy breed 🐕, is affectionate with a thick coat. Bred to guard monasteries, it’s loyal and intelligent. Tibetans need grooming and exercise, thriving as devoted companions in active families. 🙏",
    "n02097658-silky_terrier": "The Silky Terrier, a small toy breed 🐾, is bold with a fine, blue-and-tan coat. Energetic and affectionate, it loves playtime. Silkies need grooming and exercise, thriving as charming companions in active homes. 💃",
    "n02098105-soft-coated_wheaten_terrier": "The Soft-Coated Wheaten Terrier, a lively breed 🐶, is friendly with a soft, wheat-colored coat. Playful and affectionate, it loves families. Wheatens need grooming and exercise, thriving as spirited companions for active owners. 🌾",
    "n02098286-West_Highland_white_terrier": "The West Highland White Terrier, a small breed 🐾, is bold with a crisp, white coat. Spunky and affectionate, it’s energetic and loyal. Westies need grooming and exercise, thriving as charming companions in active homes. 🥰",
    "n02098413-Lhasa": "The Lhasa Apso, a Tibetan breed 🐕, is loyal with a long, flowing coat. Bred to guard monasteries, it’s alert and affectionate. Lhasas need grooming and moderate exercise, thriving as regal companions for attentive owners. 🏯",
    "n02099267-flat-coated_retriever": "The Flat-Coated Retriever, a cheerful gundog 🐶, is energetic with a sleek coat. Youthful and friendly, it loves water. Flat-Coats need exercise and minimal grooming, thriving as loyal companions in active, outdoor-loving families. 🏊‍♂️",
    "n02099429-curly-coated_retriever": "The Curly-Coated Retriever, a unique gundog 🐾, is intelligent with a curly coat. Bred for retrieving, it loves water. Independent yet affectionate, Curlies need exercise and grooming, thriving as devoted companions for active owners. 🌊",
    "n02099601-golden_retriever": "The Golden Retriever, a popular gundog 🐕, is friendly with a lush, golden coat. Intelligent and gentle, it excels in service roles. Goldens need exercise and grooming, thriving as loyal companions in active, loving homes. 🥰",
    "n02099712-Labrador_retriever": "The Labrador Retriever, a versatile gundog 🐶, is outgoing with a dense coat. Intelligent and friendly, it excels in service and family roles. Labs need exercise and minimal grooming, thriving as affectionate companions for active owners. 🏞️",
    "n02099849-Chesapeake_Bay_retriever": "The Chesapeake Bay Retriever, a rugged gundog 🐾, is loyal with a wavy coat. Bred for harsh conditions, it loves water. Independent yet affectionate, Chessies need exercise and grooming, thriving as devoted companions for active owners. 🌊",
    "n02100236-German_short-haired_pointer": "The German Shorthaired Pointer, a versatile gundog 🐕, is energetic with a liver-and-white coat. Intelligent and loyal, it excels in hunting. GSPs need ample exercise and minimal grooming, thriving as companions for active owners. 🏃‍♂️",
    "n02100583-vizsla": "The Vizsla, a Hungarian gundog 🐶, is affectionate with a rusty-gold coat. Loyal and energetic, it loves hunting and companionship. Vizslas need exercise and minimal grooming, thriving as devoted companions in active, loving homes. 🌞",
    "n02100735-English_setter": "The English Setter, an elegant gundog 🐾, is gentle with a silky coat. Bred for hunting, it’s energetic and graceful. Affectionate and loyal, English Setters need exercise and grooming, thriving as companions for active families. 🏞️",
    "n02100877-Irish_setter": "The Irish Setter, a vibrant gundog 🐕, is friendly with a stunning red coat. Energetic and elegant, it loves outdoor activities. Irish Setters need exercise and grooming, thriving as spirited companions in active households. 🔥",
    "n02101006-Gordon_setter": "The Gordon Setter, a stately gundog 🐶, is loyal with a glossy black-and-tan coat. Intelligent and enduring, it excels in hunting. Gordons need exercise and grooming, thriving as devoted companions for active, dedicated owners. 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "n02101388-Brittany_spaniel": "The Brittany Spaniel, a compact gundog 🐾, is energetic with an orange-and-white coat. Affectionate and versatile, it loves outdoor adventures. Brittanys need exercise and minimal grooming, thriving as cheerful companions for active families. 🏃‍♂️",
    "n02101556-clumber": "The Clumber Spaniel, a heavy gundog 🐶, is gentle with a dense, white coat. Loyal and calm, it’s ideal for families. Clumbers need moderate exercise and grooming, thriving as devoted, relaxed companions for attentive owners. 🛋️",
    "n02102040-English_springer": "The English Springer Spaniel, a lively gundog 🐾, is friendly with a silky coat. Known for flushing game, it loves activity. Springers need exercise and grooming, thriving as affectionate companions in active, family-oriented homes. 🏞️",
    "n02102177-Welsh_springer_spaniel": "The Welsh Springer Spaniel, a compact gundog 🐶, is loyal with a red-and-white coat. Energetic and affectionate, it loves hunting. Welshies need exercise and grooming, thriving as cheerful companions for active owners. 🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "n02102318-cocker_spaniel": "The Cocker Spaniel, a charming gundog 🐾, is affectionate with a silky coat. Its soulful eyes and playful nature shine in families. Cockers need grooming and exercise, thriving as joyful companions in loving, active homes. 🥰",
    "n02102480-Sussex_spaniel": "The Sussex Spaniel, a low-slung gundog 🐶, is gentle with a golden-liver coat. Calm and affectionate, it loves family life. Sussex Spaniels need moderate exercise and grooming, thriving as loyal companions for attentive owners. 🛋️",
    "n02102973-Irish_water_spaniel": "The Irish Water Spaniel, a unique gundog 🐾, is playful with a curly, liver coat. Loving water, it excels in retrieving. Independent yet affectionate, it needs exercise and grooming, thriving as a devoted companion. 🇮🇪",
    "n02104029-kuvasz": "The Kuvasz, a Hungarian guard dog 🛡, is loyal with a thick, white coat. Courageous and protective, it’s a strong guardian. Kuvaszok need grooming and exercise, thriving as devoted companions for experienced owners. 🏰",
    "n02104365-schipperke": "The Schipperke, a Belgian herding dog 🐶, is alert with a fox-like face and black coat. Agile and curious, it’s a great watchdog. Schipperkes need exercise and grooming, thriving as spirited companions in active homes. 🦊",
    "n02105056-groenendael": "The Groenendael, a Belgian Shepherd 🐕, is intelligent with a long, black coat. Versatile and loyal, it excels in protection. Groenendaels need exercise and grooming, thriving as elegant, devoted companions for active owners. 🛡",
    "n02105162-malinois": "The Malinois, a Belgian Shepherd 🐶, is alert with a short, fawn coat. Intelligent and loyal, it excels in police work. Malinois need ample exercise and training, thriving as devoted companions for active owners. 🚨",
    "n02105251-briard": "The Briard, a French herding dog 🐾, is loyal with a shaggy coat. Protective and intelligent, it loves families. Briards need grooming and exercise, thriving as devoted companions for active owners who value their charm. 🐑",
    "n02105412-kelpie": "The Australian Kelpie, a tireless herder 🐶, is energetic with a short coat. Intelligent and loyal, it excels in livestock work. Kelpies need ample exercise and stimulation, thriving as devoted companions for active owners. 🇦🇺",
    "n02105505-komondor": "The Komondor, a Hungarian guard dog 🛡, is loyal with a corded coat. Courageous and protective, it’s a strong guardian. Komondors need grooming and exercise, thriving as devoted companions for experienced owners. 🐑",
    "n02105641-Old_English_sheepdog": "The Old English Sheepdog, a shaggy herder 🐶, is gentle with a fluffy coat. Intelligent and playful, it loves families. Sheepdogs need grooming and exercise, thriving as affectionate companions in active homes. 🥰",
    "n02105855-Shetland_sheepdog": "The Shetland Sheepdog, a small herder 🐾, is intelligent with a long coat. Agile and loyal, it resembles a mini Collie. Shelties need grooming and exercise, thriving as devoted companions for active owners. 🐑",
    "n02106030-collie": "The Collie, a Scottish herding dog 🐶, is intelligent with a lush coat. Gentle and loyal, it loves families. Collies need grooming and exercise, thriving as affectionate companions for owners valuing their elegance. 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "n02106166-Border_collie": "The Border Collie, a brilliant herder 🐾, is energetic with a medium coat. Intelligent and agile, it excels in tasks. Borders need exercise and stimulation, thriving as devoted companions for active owners. 🏃‍♂️",
    "n02106382-Bouvier_des_Flandres": "The Bouvier des Flandres, a Belgian herder 🐶, is loyal with a shaggy coat. Strong and calm, it’s a great guardian. Bouviers need grooming and exercise, thriving as devoted companions for active owners. 🛡",
    "n02106550-Rottweiler": "The Rottweiler, a German guard dog 🦾, is loyal with a short, black-and-tan coat. Confident and protective, it’s affectionate with family. Rotties need exercise and training, thriving as devoted companions for experienced owners. 🛡",
    "n02106662-German_shepherd": "The German Shepherd, a versatile worker 🐕, is intelligent with a medium coat. Loyal and protective, it excels in service roles. Shepherds need exercise and training, thriving as reliable companions for active owners. 🚨",
    "n02107142-Doberman": "The Doberman Pinscher, a sleek guard dog 🐶, is loyal with a short coat. Intelligent and protective, it’s affectionate with family. Dobermans need exercise and training, thriving as devoted companions for active owners. 🛡",
    "n02107312-miniature_pinscher": "The Miniature Pinscher, a tiny toy breed 🐾, is fearless with a sleek coat. Bold and energetic, it loves attention. Min Pins need moderate exercise and grooming, thriving as spirited companions in active homes. 👑",
    "n02107574-Greater_Swiss_Mountain_dog": "The Greater Swiss Mountain Dog, a strong worker 🦾, is loyal with a tricolor coat. Gentle and calm, it loves families. Swissies need exercise and grooming, thriving as devoted companions for active owners. 🏔️",
    "n02107683-Bernese_mountain_dog": "The Bernese Mountain Dog, a Swiss giant 🐶, is loyal with a tricolor coat. Calm and affectionate, it loves families. Berners need grooming and exercise, thriving as devoted companions in cold climates. ❄️",
    "n02107908-Appenzeller": "The Appenzeller Sennenhund, a Swiss worker 🐾, is energetic with a tricolor coat. Loyal and driven, it excels in herding. Appenzellers need exercise and grooming, thriving as devoted companions for active owners. 🏔️",
    "n02108000-EntleBucher": "The Entlebucher Mountain Dog, a compact worker 🐶, is loyal with a tricolor coat. Energetic and cheerful, it loves families. Entlebuchers need exercise and grooming, thriving as devoted companions for active owners. 🏔️",
    "n02108089-boxer": "The Boxer, a German working dog 🦾, is playful with a short, fawn coat. Loyal and protective, it loves families. Boxers need exercise and minimal grooming, thriving as affectionate companions in active homes. 🥊",
    "n02108422-bull_mastiff": "The Bull Mastiff, a British guard dog 🛡, is loyal with a short coat. Gentle and protective, it’s affectionate with family. Bull Mastiffs need moderate exercise and grooming, thriving as devoted companions for calm owners. 🏡",
    "n02108551-Tibetan_mastiff": "The Tibetan Mastiff, a massive guard dog 🦾, is loyal with a thick coat. Protective and independent, it’s calm with family. Tibetans need grooming and exercise, thriving as devoted companions in spacious homes. 🏯",
    "n02108915-French_bulldog": "The French Bulldog, a compact companion 🐶, is affectionate with a smooth coat. Playful and charming, it loves lounging. Frenchies need minimal grooming and exercise, thriving as loyal companions in relaxed homes. 🛋️",
    "n02109047-Great_Dane": "The Great Dane, a German giant 🐕, is friendly with a short coat. Loyal and calm, it loves families. Danes need moderate exercise and grooming, thriving as affectionate companions in spacious homes. 🏰",
    "n02109525-Saint_Bernard": "The Saint Bernard, a Swiss rescue dog 🐶, is gentle with a thick coat. Loyal and calm, it loves families. Saints need grooming and moderate exercise, thriving as devoted companions in cold climates. ❄️",
    "n02109961-Eskimo_dog": "The Eskimo Dog, a Canadian sled dog 🛷, is loyal with a thick, white coat. Strong and energetic, it thrives in cold climates. Eskimos need exercise and grooming, thriving as devoted companions for active owners. ❄️",
    "n02110063-malamute": "The Alaskan Malamute, a powerful sled dog 🛷, is loyal with a fluffy coat. Strong and friendly, it loves outdoor work. Malamutes need exercise and grooming, thriving as devoted companions in cold climates. ❄️",
    "n02110185-Siberian_husky": "The Siberian Husky, a sleek sled dog 🛷, is friendly with a thick coat. Energetic and striking, it loves adventure. Huskies need exercise and grooming, thriving as devoted companions for active owners in cold climates. ❄️",
    "n02110627-affenpinscher": "The Affenpinscher, a toy breed 🐾, is playful with a wiry, monkey-like face. Bold and loyal, it loves attention. Affens need grooming and moderate exercise, thriving as charming companions in active homes. 🐒",
    "n02110806-basenji": "The Basenji, an African hound 🐶, is quiet and intelligent with a sleek coat. Barkless and cat-like, it’s affectionate and independent. Basenjis need exercise and minimal grooming, thriving as unique companions for active owners. 🌍",
    "n02110958-pug": "The Pug, a Chinese toy breed 🐾, is affectionate with a wrinkled face and short coat. Playful and clownish, it loves lounging. Pugs need minimal grooming and exercise, thriving as loyal companions in relaxed homes. 😆",
    "n02111129-Leonberg": "The Leonberger, a German giant 🦁, is loyal with a thick, lion-like mane. Calm and affectionate, it loves families. Leos need grooming and exercise, thriving as devoted companions in spacious, active homes. 🏰",
    "n02111277-Newfoundland": "The Newfoundland, a Canadian working dog 🐶, is gentle with a thick coat. Loyal and calm, it excels in water rescue. Newfies need grooming and exercise, thriving as affectionate companions in spacious homes. 🌊",
    "n02111500-Great_Pyrenees": "The Great Pyrenees, a French guard dog 🛡, is loyal with a thick, white coat. Calm and protective, it loves families. Pyrs need grooming and exercise, thriving as devoted companions in spacious, cold climates. ❄️",
    "n02111889-Samoyed": "The Samoyed, a Siberian sled dog 🛷, is friendly with a stunning white coat. Gentle with a ‘Sammy smile,’ it loves families. Sammies need grooming and exercise, thriving as affectionate companions in cold climates. ❄️",
    "n02112018-Pomeranian": "The Pomeranian, a tiny toy breed 🐾, is lively with a fluffy coat. Bold and fox-like, it loves attention. Poms need grooming and moderate exercise, thriving as charming companions in active, loving homes. 💖",
    "n02112137-chow": "The Chow Chow, a Chinese breed 🦁, is loyal with a lion-like mane. Independent yet affectionate, it’s a strong guardian. Chows need grooming and exercise, thriving as devoted companions for experienced owners. 🏯",
    "n02113023-Pembroke": "The Pembroke Welsh Corgi, a small herding dog 🐶, is intelligent and loyal with a short, red-and-white coat. Known for its short legs and big personality, it loves activity. Pembrokes need moderate exercise and grooming, thriving as devoted companions for active, family-oriented owners. 🏴",
    "n02113186-Cardigan": "The Cardigan Welsh Corgi, a sturdy herder 🐾, is loyal and intelligent with a short, multicolored coat and long tail. Known for its work ethic and affectionate nature, it loves families. Cardigans need moderate exercise and grooming, thriving as devoted companions for active owners. 🏴",
    "n02113624-toy_poodle": "The Toy Poodle, a tiny companion 🐶, is intelligent and playful with a curly, dense coat. Known for its elegance and agility, it loves attention. Toy Poodles need regular grooming and moderate exercise, thriving as charming, devoted companions in active, loving households. 💃",
    "n02113712-miniature_poodle": "The Miniature Poodle, a small companion 🐾, is smart and lively with a curly, dense coat. Known for its versatility and elegance, it excels in agility. Minis need regular grooming and exercise, thriving as affectionate, devoted companions for active owners who value their charm. 🏆",
    "n02113799-standard_poodle": "The Standard Poodle, a large companion 🐶, is intelligent and elegant with a curly, dense coat. Known for its versatility and athleticism, it excels in various roles. Standards need regular grooming and exercise, thriving as loyal, devoted companions for active, dedicated owners. 🌟",
    "n02113978-Mexican_hairless": "The Mexican Hairless (Xoloitzcuintli), a unique breed 🐕, is loyal and intelligent with a smooth, hairless body. Known for its ancient heritage and calm nature, it’s affectionate with family. Xolos need minimal grooming and moderate exercise, thriving as devoted companions for attentive owners. 🌵",
    "n02115641-dingo": "The Dingo, a wild Australian dog 🐺, is independent and agile with a lean, sandy coat. Known for its survival instincts and elusive nature, it’s not a typical pet. Dingoes need space and experienced handlers, thriving in wild-like environments rather than domestic homes. 🌏",
    "n02115913-dhole": "The Dhole, an Asian wild dog 🦊, is social and fast with a reddish-brown coat. Known for its pack-hunting skills and vocal nature, it’s not a domestic pet. Dholes thrive in wild environments, requiring vast spaces and are unsuitable for typical pet ownership. 🌍",
    "n02116738-African_hunting_dog": "The African Hunting Dog, a wild pack hunter 🐆, is swift and social with a mottled, multicolored coat. Known for its endurance and cooperative hunting, it’s not a pet. African Hunting Dogs thrive in wild environments, requiring vast spaces and are unsuitable for domestic life. 🦒"
}

st.title("🐶Dog Breed Identification App")
st.write("💻Upload a dog image and the app will predict its breed🐕‍🦺.")

# Option selector
option = st.radio(
    "Choose input method:",
    ("Upload Image", "Web Capture")
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
elif option == "Web Capture":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')

if image is not None:
    st.image(image, caption='Input Image', use_container_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    max_prob = float(np.max(preds))
    breed_idx = int(np.argmax(preds))
    breed = idx_to_breed[breed_idx]
    confidence_threshold = 0.5  # You can adjust this value

    if max_prob < confidence_threshold:
        st.error("This is not a dog or the breed is not recognized.")
    else:
        st.write(f"Predicted Breed: **{breed}** (Confidence: {max_prob:.2f})")
        description = breed_descriptions.get(breed, "Description not available for this breed.")
        st.info(description)