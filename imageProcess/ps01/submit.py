import argparse
from builtins import input

from nelson.gtomscs import submit

LATE_POLICY = """Late Assignments Policy:
  \"I have read the late assignments policy for CS6476. I understand that 
  only my last commit before the deadline will be accepted.\"
"""

HONOR_PLEDGE = "Honor Pledge:\n\n  \"I have neither given nor received aid " \
               "on this assignment.\"\n"


def require_pledges():
    print(LATE_POLICY)
    ans = input("Please type 'yes' to agree and continue>")
    if ans != "yes":
        raise RuntimeError("Late Assignments policy not accepted.")

    print()
    print(HONOR_PLEDGE)
    ans = input("Please type 'yes' to agree and continue>")
    if ans != "yes":
        raise RuntimeError("Honor pledge not accepted")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Submits code to the Udacity site.')
    parser.add_argument('part', choices=['ps01', 'ps01_report'])
    args = parser.parse_args()

    quiz = args.part
    course = "cs6476"

    if quiz == "ps01":
        filenames = ["ps1.py"]
    else:
        filenames = ['ps1_report.pdf', 'experiment.py', 'ps1-1-a-1.png', 'ps1-1-a-2.png']

    require_pledges()

    submit(course, quiz, filenames)

if __name__ == "__main__":
    main()
