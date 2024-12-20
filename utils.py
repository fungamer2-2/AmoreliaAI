def num_to_str_sign(val, num_dec):
	assert isinstance(num_dec, int) and num_dec > 0
	val = round(val, num_dec)
	if val == -0.0:
		val = 0.0
	
	sign = "+" if val >= 0 else ""
	f = "{:." + str(num_dec) + "f}"
	return sign + f.format(val)


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
