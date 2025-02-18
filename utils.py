import re, os
from colored import Style


def clear_screen():
	os.system("cls" if os.name == "nt" else "clear")


def num_to_str_sign(val, num_dec):
	assert isinstance(num_dec, int) and num_dec > 0
	val = round(val, num_dec)
	if val == -0.0:
		val = 0.0
	
	sign = "+" if val >= 0 else ""
	f = "{:." + str(num_dec) + "f}"
	return sign + f.format(val)
	

def val_to_symbol_color(val, maxsize, color_pos="", color_neg="", val_scale=1.0):
	bars = round(abs(val / val_scale) * maxsize)
	if bars == 0:
		return "="
	if val >= 0:
		return color_pos + "+"*bars + Style.reset
	else:
		return color_neg + "-"*bars + Style.reset


def get_approx_time_ago_str(timedelta):
	secs = int(timedelta.total_seconds())
	if secs < 60:
		return "just now"		
	minutes = secs // 60
	if minutes < 60:
		return f"{minutes} minutes ago"
	hours = minutes // 60
	if hours < 24:
		return f"{hours} hours ago"
	days = hours // 24
	return f"{days} days ago"


def normalize_text(text):			
	text = text.lower()
	for symbol in ".,:;!?":
		text = text.replace(symbol, " ")
	
	text = " ".join(text.split())
	text = text.replace("’", "'")
	
	contractions = {
		"here's": "here is",
		"there's": "there is",
		"can't": "cannot",
		"don't": "do not",
		"doesn't": "does not",
		"didn't": "did not",
		"isn't": "is not",
		"aren't": "are not",
		"wasn't": "was not",
		"hasn't": "has not",
		"hadn't": "had not",
		"shouldn't": "should not",	
		"won't": "will not",
		"i'm": "i am",
		"you're": "you are",
		"we're": "we are",
		"they're": "they are",
		"i've": "i have",
		"you've": "you have",
		"we've": "we have",
		"they've": "they have",
		"y'all": "you all",	
		"that's": "that is",
		"it's": "it is",
		"it'd": "it would",
		"i'll": "i will",
		"you'll": "you will",
		"he'll": "he will",
		"she'll": "she will",
		"we'll": "we will",
		"they'll": "they will",
		"gonna": "going to",
		"could've": "could have",
		"should've": "should have",
		"would've": "would have",
		"gimme": "give me",
		"gotta": "got to",
		"how's": "how is",
	}
	def _replacement(match):
		bound1 = match.group(1)
		txt = match.group(2)
		bound2 = match.group(3)
		return f"{bound1}{contractions[txt]}{bound2}"
	
	for c in contractions:
		text = re.sub(rf"(\b)({c})(\b)", _replacement, text)
	return text


def conversation_to_string(messages, ai_name="AI"):	
	role_map = {
		"user": "User",
		"assistant": ai_name
	}
	return "\n\n".join(
		f"{role_map[msg['role']]}: {msg['content']}"
		for msg in messages[:-1]
	)
	