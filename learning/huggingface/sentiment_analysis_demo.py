from transformers import pipeline


summarizer = pipeline("summarization")
data = "Transformers (stylised as TRANSFORMERS, alternatively titled as TransFormers, or simply abbreviated TF), is a media franchise produced by Japanese toy company Takara Tomy and American toy company Hasbro. It primarily follows the heroic Autobots and the villainous Decepticons, two alien robot factions at war that can transform into other forms, such as vehicles and animals. The franchise encompasses toys, animation, comic books, video games and films. As of 2011, it generated more than ¥2 trillion ($25 billion) in revenue,[1] making it one of the highest-grossing media franchises of all time."

summarizer(data)
