#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Supervised Learning: Comparing Trainable Classifiers']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Final Project in CSE 415, University of Washington, Winter, 2018
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Trainable_Classifiers_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:


Brandon = Partner("TerLouw", "Brandon", "bterlouw")
Warren = Partner("Cho", "Warren", "wcho")
team = [Brandon, Warren]

OPTION = '3'
# Legal options are 1, 2, and 3.

title = "A revolutionary classifier"
 # In this case, the Python file for the formulation would be named End_Poverty.py.

approach = '''Our approach will be to first understand how
the different classifiers work, choose which we will implement,
code two different classifiers, train each with two different data sets,
compare how each did, and then optimize using either bagging
or boosting.'''

workload_distribution = '''We have a general idea of which direction we
want to go for this project, two-layer feedforward neural networks. But we
will start by exploring all options. Becuase we will have two classifiers,
we can either develop one each in parallel in their entirety or both
develop the same parts for each classifier. Initial tickets to be issued
would be classification, regression, regularization, algorithm, and output.
We will develop with the intent to run initial tests with the CIFAR10
dataset.'''

reference1 = '''Feedforward Neural Networks by McGonagle;
    URL: https://brilliant.org/wiki/feedforward-neural-networks/ (accessed Feb. 26, 2018)'''

reference2 = '''"Introduction to Random Forest Algorithm" by Saimadhu Polamuri,
    source: http://dataaspirant.com/2017/05/22/random-forest-algorithm-machine-learing/ (accessed 26 Feb 2018)'''

our_submission = Who_and_what(team, OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())