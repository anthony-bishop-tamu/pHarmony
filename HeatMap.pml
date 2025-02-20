fetch 4DEP
remove chain A+B+C
create IL1B_obj, chain D
color grey, IL1B_obj
center IL1B_obj
show surface, IL1B_obj
remove 4DEP and chain D
alter all, b=0
rebuild
select CSPs, resid "39+69+85+119+132+139+140+142+145 and IL1B_obj
alter resi 39 and IL1B_obj, b=1.000000
alter resi 69 and IL1B_obj, b=1.000000
alter resi 85 and IL1B_obj, b=1.000000
alter resi 119 and IL1B_obj, b=1.000000
alter resi 132 and IL1B_obj, b=1.000000
alter resi 139 and IL1B_obj, b=1.000000
alter resi 140 and IL1B_obj, b=1.000000
alter resi 142 and IL1B_obj, b=1.000000
alter resi 145 and IL1B_obj, b=1.000000
spectrum b, blue_red, CSPs
color green, 4DEP
