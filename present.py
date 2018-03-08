import MultLayerCall as m
import RandomForest as w
from pip._vendor.distlib.compat import raw_input

a = str(raw_input("Which classifier would you like to use? Neural Net?(n) or Random Forest?(f)"))
if(a=='n'):
    m.run()
elif(a=='r'):
    w.run()
else:
    print("Sorry that is not an option.")