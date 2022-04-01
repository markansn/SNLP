import torch
import time
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datasets import load_dataset
# tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
# #
# # model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")


def get_pretrained(name):
  tokenizer = AutoTokenizer.from_pretrained(name)

  model = AutoModelForSeq2SeqLM.from_pretrained(name)

  return model, tokenizer


# article = """
# Justin Timberlake and Jessica Biel, welcome to parenthood.
# The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
# "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
# The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
# """

# data = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="test[:1%]")

# inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)

def do_generation(article, model, tokenizer, max_length, min_length):
    inputs = tokenizer.encode(article, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length)
    # length_penalty=2.0,
    # num_beams=4,
    # early_stopping=True)
    # just for debugging
    # print(tokenizer.decode(outputs))
    # print(outputs)
    return outputs

articles = [
            {'article': '(CNN)SPOILER ALERT! It\'s not just women getting cloned. That was the big twist at the end of "Orphan Black\'s" second season. The kickoff to the new season leads the list of six things to watch in the week ahead. 1. "Orphan Black," 9 p.m. ET, Saturday, April 18, BBC America . The cloning cult sci-fi series remains one of the most critically acclaimed shows on TV, thanks in large part to the performance of Tatiana Maslany, who has taken on at least six roles on the show so far, including a newly introduced transgender clone. Maslany told reporters this week that we can expect even more impressive scenes with multiple clones. "We like to push the boundaries of what we\'re able to do and the limits of those clone scenes," she said. "So, yes, you\'ll definitely see more complex clone work this season and that\'s just because we\'re getting more comfortable with the technology and we\'re excited by getting to sort of further complicate things." And the introduction of a group of male clones will certainly increase the suspense. "There definitely is a shift towards the Castor clones that we get to explore them a little bit more," she said. The fans of the show, dubbed the "Clone Club" have a lot to look forward to when the show premieres on Saturday the 18th, and Maslany is blown away by the response to the series so far. "We\'ve always been really humbled and really inspired by our fans and by their dedication to the show and their knowledge of the show, and  just how it changes their own lives. It\'s incredible." 2. "Turn: Washington\'s Spies," 9 p.m. ET, Monday, AMC . The series about spies in the early days of the Revolutionary War returns with a new subtitle, "Washington\'s Spies," and a new Monday night time slot. Series star Jamie Bell told CNN what we can expect in the second season. "This year we have a lot more battles; we have the journey of [George] Washington and we\'re getting under his skin a little bit as well. We also introduce new characters like Benedict Arnold, a very infamous character in American history." Bell hopes the series might bring more recognition to the Culper spy ring and everything it did. "I think there should be a monument to all of the Culper ring somewhere. I was amazed that there is nothing [in Washington] about these people who did something extraordinary." 3. "Game of Thrones," 9 p.m. ET, Sunday, HBO . The world of Westeros returns for a fifth season in one of the biggest season premieres of the year. Click here for more on what to expect. 4. "Justified," 10 p.m. ET, Tuesday, FX . Timothy Olyphant\'s tour de force performance as Raylan Givens comes to an end Tuesday night, as the modern-day Western airs its season finale. We\'ll have to see how his final showdown with Boyd Crowder goes. 5. "Veep," 10:30 ET, Sunday, HBO . Hugh Laurie joins the cast and Julia Louis-Dreyfus is now the president of the United States on HBO\'s hit comedy. 6. "Nurse Jackie," 9 p.m. ET, Sunday, Showtime . The final season of Showtime\'s long-running melodrama begins.', 'highlights': 'Critically acclaimed series "Orphan Black" returns .\n"Turn: Washington\'s Spies" starts a second season .\n"Game of Thrones" is back for season five .', 'id': '00dddbedf41ec993a8b976f3cce2dd8ca2c7efed'},
            {'article': '(CNN)Emergency operators get lots of crazy calls, but few start like this. Caller:  "Hello, I\'m trapped in this plane and I called my job, but I\'m in this plane." Operator:  "You\'re where?" Caller:  "I\'m inside a plane and I feel like it\'s up moving in the air.  Flight 448 can you please tell somebody (to) stop it." The frantic 911 call came just as the Alaska Airlines flight had taken off from Seattle-Tacoma International Airport on Monday afternoon.  The caller was a ramp agent who fell asleep in the plane\'s cargo hold. The cell phone call soon broke up, but the man was making himself known in other ways as the crew and passengers reported unusual banging from the belly of the Boeing 737. The pilot radioed air traffic control and said he would make an emergency landing. "There could be a person in there so we\'re going to come back around," he told air traffic control. The ramp agent who took the untimely nap and caused all the fuss is an employee of Menzies Aviation, a contractor for Alaska Airlines that handles loading the luggage. He\'ll no longer have the option of dozing aboard one of the airline\'s planes. "The Menzies employee has been permanently banned from working on Alaska Airlines planes," said Bobbie Egan, a spokeswoman for the airline. Flight 448, which was on its way to Los Angeles, only spent 14 minutes in the air. Other than being scared, the agent never was in any real danger. The cargo hold is pressurized and temperature controlled, the airline said. The passengers knew something wasn\'t right, almost as soon as the plane took off. "All of a sudden we heard all this pounding underneath the plane and we thought there was something wrong with the landing gear," Robert Higgins told CNN affiliate KABC. Not everyone heard the banging, but it was soon clear this wasn\'t a normal flight. "We just took off for L.A. regular and then ... about five minutes into the flight the captain came on and said we were going back and we\'d land within five to seven minutes, and we did," passenger Marty Collins told affiliate KOMO. "When we landed was when all the trucks and the police and the fire trucks surrounded the plane." "I think it\'s scary and really unsafe, too," Chelsie Nieto told affiliate KCPQ. "Because what if it\'s someone who could have been a terrorist?" The employee started work at 5 a.m. and his shift was scheduled to end at 2:30 p.m., just before the flight departed. The agent was off the two days prior to the incident and had taken a lunch break and a break in the afternoon before making his way into the cargo hold, according to a source familiar with the investigation. The man had been on a four-person team loading baggage onto the flight. "During a pre-departure huddle, the team lead noticed the employee was missing. The team lead called into the cargo hold for the employee and called and texted the employee\'s cell phone, but did not receive an answer. His co-workers believed he finished his shift and went home," the airline\'s blog said. It\'s believed he was hidden by luggage, making it difficult for the rest of his team to see him, the source said. All ramp employees have security badges, and undergo full criminal background checks before being hired, according to the airline. After the delay, the flight with 170 passengers and six crew members on board made it to Los Angeles a couple of hours late. CNN\'s Dave Alsup, Joshua Gaynor and Greg Morrison contributed to this report.', 'highlights': "The ramp agent fell asleep in the plane's cargo hold .\nHe can no longer work on Alaska Airlines flights .", 'id': '00e5eb6c1af59233b661e59b0954a4f84a2f3904'},
            {'article': '(CNN)Wanted: film director, must be eager to shoot footage of golden lassos and invisible jets. CNN confirms that Michelle MacLaren is leaving the upcoming "Wonder Woman" movie (The Hollywood Reporter first broke the story). MacLaren was announced as director of the movie in November. CNN obtained a statement from Warner Bros. Pictures that says, "Given creative differences, Warner Bros. and Michelle MacLaren have decided not to move forward with plans to develop and direct \'Wonder Woman\' together." (CNN and Warner Bros. Pictures are both owned by Time Warner.) The movie, starring Gal Gadot in the title role of the Amazon princess, is still set for release on June 23, 2017. It\'s the first theatrical movie centering around the most popular female superhero. Gadot will appear beforehand in "Batman v. Superman: Dawn of Justice," due out March 25, 2016. In the meantime, Warner will need to find someone new for the director\'s chair.', 'highlights': 'Michelle MacLaren is no longer set to direct the first "Wonder Woman" theatrical movie .\nMacLaren left the project over "creative differences"\nMovie is currently set for 2017 .', 'id': '027936fedb5e785fe79e84cb6e55c9cc26042ad3'},
            {'article': '(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.', 'highlights': 'James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .', 'id': '00200e794fa41d3f7ce92cbf43e9fd4cd652bb09'},
            {'article': '(CNN)The attorney for a suburban New York cardiologist charged in what authorities say was a failed scheme to have another physician hurt or killed is calling the allegations against his client "completely unsubstantiated." Appearing Saturday morning on CNN\'s "New Day," Randy Zelin defended his client, Dr. Anthony Moschetto, who faces criminal solicitation, conspiracy, burglary, arson, criminal prescription sale and weapons charges in connection to what prosecutors called a plot to take out a rival doctor on Long Island. "None of anything in this case has any evidentiary value," Zelin told CNN\'s Christi Paul.  "It doesn\'t matter what anyone says, he is presumed to be innocent." Moschetto,54, pleaded not guilty to all charges Wednesday.  He was released after posting $2 million bond and surrendering his passport. Zelin said that his next move is to get Dr. Moshetto back to work. "He\'s got patients to see. This man, while he was in a detention cell, the only thing that he cared about were his patients. And amazingly, his patients were flooding the office with calls, making sure that he was OK," Zelin said. Two other men -- identified as James Chmela, 43, and James Kalamaras, 41 -- were named as accomplices, according to prosecutors. They pleaded not guilty in Nassau County District Court, according to authorities. Both were released on bail. A requests for comment from an attorney representing Chmela was not returned. It\'s unclear whether Kalamaras has retained an attorney. Police officers allegedly discovered approximately 100 weapons at Moschetto\'s home, including hand grenades, high-capacity magazines and knives. Many of the weapons were found in a hidden room behind a switch-activated bookshelf, according to prosecutors. The investigation began back in December, when undercover officers began buying heroin and oxycodone pills from Moschetto in what was initially a routine investigation into the sale of prescription drugs, officials said. During the course of the undercover operation, however, Moschetto also sold the officers two semiautomatic assault weapons as well as ammunition, prosecutors said. Moschetto allegedly told officers during one buy that he needed dynamite to "blow up a building." He later said he no longer needed the dynamite because a friend was setting fire to the building instead. Kalamaras and Chmela are believed to have taken part in the arson, according to prosecutors. "The fire damaged but did not destroy the office of another cardiologist whose relationship with Dr. Moschetto had soured due to a professional dispute," according to the statement from the district attorney\'s office. Moschetto allegedly gave an informant and undercover detective blank prescriptions and cash for the assault and killing of the fellow cardiologist, according to prosecutors. He also requested that the rival\'s wife be assaulted if she happened to be present, authorities said. "He was willing to pay $5,000 to have him beaten and put in a hospital for a few months, and then he said he would pay $20,000 to have him killed," said Assistant District Attorney Anne Donnelly, according to CNN affiliate WCBS.', 'highlights': 'A lawyer for Dr. Anthony Moschetto says the charges against him are baseless .\nMoschetto, 54, was arrested for selling drugs and weapons, prosecutors say .\nAuthorities allege Moschetto hired accomplices to burn down the practice of former associate .', 'id': '0021fe8d65bd0d6d76d5fefba2ac02f0c48a43f4'}
]
def get_article(file):
    with open(file, 'r') as f:
        return f.read()

# "ainize/bart-base-cnn" "facebook/bart-large-cnn" "flax-commun ity/t5-base-cnn-dm"
model, tokenizer = get_pretrained("sshleifer/distilbart-cnn-6-6")
print(tokenizer.vocab_size)
# print(len(data))
# print(data[2])
for i in range(0, 5):
  article = articles[i]['article'][:511]
  summary = articles[i]['highlights']
  total_times = 0
  min_val = float('inf')
  max_val = 0
  try:
    for j in tqdm(range(0, 10)):
      start_time = time.time()
      outputs = do_generation(article, model, tokenizer, 100, 40)
      stop_time = time.time() - start_time
      total_times += stop_time
      if stop_time > max_val:
        max_val = stop_time
      elif stop_time < min_val:
        min_val = stop_time
    variance = max_val - min_val
    avg = total_times / 10

    print("avg " + str(i) + " " + str(avg))
    print("var " + str(i) + " " + str(variance))
  except:
    print("FAIL")


# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# rouge = Rouge()

# print(rouge.get_scores(tokenizer.decode(outputs[0]), summary))

# print(get_article("article1.txt"))