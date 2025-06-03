from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from carousel import Carousel

def load_dataset(data):
    """
    This function reads the csv file and stores it in nested list
    Input: str
    Returns: nested list
    """
    with open(data, 'r') as file:
      lines = file.read().splitlines()
      for i in range (len(lines)):
        lines[i] = lines[i].split(",") #  Nested list by commas
      return lines

def remove_blank_helper(data, index):
    """
    This function checks for blank entries at a specific column index then returns
    only the rows that are not blank
    Input: list (Nested List)
    Returns: list (Nested List Refined. Gets rid blank entry rows at a specific column index)
    """
    abc = "abcdefghijklmnopqrstuvwxyz"
    data_blank = []
    missing_count = 0

    # Checks for blank entries
    for row in data[1:]:
      if row[index] != '':
        data_blank.append(row)
      else:
        missing_count += 1

    # Prints only if there are blank entries in the column called
    if missing_count > 0:
      print(f"\tColumn {abc[index]}: {missing_count} values missing")

    return data_blank

def remove_blank(data):
    """
    This function recursivly calls the helper function at each column index
    and overwrites the old data with new data free of blank entries at that column
    Input: list (Nested List)
    Returns: list (Nested List Refined. Gets rid of all blank entry rows)
    """

    # loops through each column index and recursively calls the helper function
    for column in range(len(data[0])):
      data = remove_blank_helper(data, column)

    return data

def remove_age_ninety(data):
    """
    This function removes rows that have age > 90
    Input: list (Nested List)
    Returns: list (Nested List Refined. Gets rid of all rows with age > 90)
    """
    data_oldies = []
    oldies_count = 0

    # Creates list of entries with age < 90
    for row in data[1:]:
      if int(row[0]) < 90:
        data_oldies.append(row)
      else:
        oldies_count += 1
    
    print(f"\tNumber of records with age > 90: {oldies_count}")
    return data_oldies

def histo_(data):
    """
    This function creates two histogram of the age of borrowers
    One for borrowers in default and one for borrowers not in default
    Input: list (Nested List)
    Returns: None
    """
    in_default_data = []
    not_in_default_data = []
    labels = [0, 20, 40, 60, 80, 100]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # [customes with historical defaults] [customers without historical defaults]
    for row in data:
      if row[2] == "OWN":
        if row[10] == "Y":
          in_default_data.append(int(row[0]))
        elif row[10] == "N":
          not_in_default_data.append(int(row[0]))

    # Histogram formatting
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize = (7,9))
    axs[0].set_title("Loans in Default")
    axs[0].set_xlabel("Age(in years)")
    axs[0].set_ylabel("No. of Borrowers")
    axs[0].hist(in_default_data, bins=bins, edgecolor='black', label=labels)
    axs[1].set_title("Loans Not in Default")
    axs[1].set_xlabel("Age(in years)")
    axs[1].set_ylabel("No. of Borrowers")
    axs[1].hist(not_in_default_data, bins=bins, edgecolor='black', label=labels)
    plt.show()

def piechart_(data):
    """
    This function creates a pie chart of the defaulted and not defaulted borrowers
    Input: list (Nested List)
    Returns: None
    """
    in_default_data = []
    not_in_default_data = []

    # [customes with historical defaults] [customers without historical defaults]
    for row in data:
      if row[2] == "OWN":
        if row[10] == "Y":
          in_default_data.append(int(row[0]))
        elif row[10] == "N":
          not_in_default_data.append(int(row[0]))

    # Pie chart formatting
    percent =[len(in_default_data), len(not_in_default_data)]
    labels = 'Defaulted', 'Not Defaulted'
    fig, ax = plt.subplots()
    ax.set_title("Homeowners: Default vs. Not Default")
    ax.pie(percent, labels=labels, autopct='%1.1f%%')
    plt.show()

def balance_(data):
    """
    This function creates 2 list of borrowers
    One for borrowers who have a historical default and one for borrowers who do not
    and prints the number of borrowers in each list
    Input: list (Nested List)
    Returns: None
    """
    in_default_data = []
    not_in_default_data = []

    # [customes with historical defaults] [customers without historical defaults]
    for row in data:
      if row[2] == "OWN":
        if row[10] == "Y":
          in_default_data.append(int(row[0]))
        elif row[10] == "N":
          not_in_default_data.append(int(row[0]))

    print(f"\tNumber of borrowers who defaulted: {len(in_default_data)}")
    print(f"\tNumber of borrowers who did not default: {len(not_in_default_data)}")

def data_prep(data):
    """
    This function preps data for model training
    Input: list (Nested List)
    Returns: list (Nested List Refined)
    """
    filtered_data = []

    # [amount, income, credit history, loan status]
    for row in data[1:]:
      if row[2] == "OWN":
        filtered_data.append([row[6], row[1], row[11], row[8]])
    return filtered_data

def scale_data(data):
    """
    This function scales the data using StandardScaler
    Input: list (Nested List)
    Returns: list (Nested List Scaled)
    """
    filtered_data = []

    # [amount, income]
    for row in data:
      filtered_data.append([row[0], row[1]])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)
    return scaled_data

def data_features(data1, data2):
    """
    This function preps x data for model training
    Input: list (Nested List)
    Returns: list (Nested List. 3 columns only)
    """
    data_features = []

    # [credit history, amount, income]
    for i in range(len(data2)):
      data_features.append([data1[i][2],data2[i][0],data2[i][1]])

    return data_features

def data_labels(data):
    """
    This function preps y data for model training
    Input: list (Nested List)
    Returns: list (Nested List 1s and 0s only)
    """
    data_labels = []

    # [loan status]
    for row in data:
      data_labels.append(row[3])

    return data_labels

def request_data_prep(data):
    """
    This function takes request data and filters it to only include 
    the columns needed for prediction
    Input: nested list
    Returns: nested list
    """
    filtered_data = []

    # [amount, income, credit history]
    for row in data[1:]:
        filtered_data.append([row[7], row[2], row[11]])

    return filtered_data

def build_dictionary(raw_request_data, request_pred):
    """
    This function builds the customer profile dictionary and adds to 
    doublely linked list
    Input: 2 nested lists
    Returns: None
    """
    i = 0

    # Profile dictionary builder
    for row in raw_request_data[1:]:
      request_dictionary={}
      request_dictionary["Borrower"] = row[0]
      request_dictionary["Age"] = row[1]
      request_dictionary["Income"] = row[2]
      request_dictionary["Home_ownership"] = row[3]
      request_dictionary["Employment"] = row[4]
      request_dictionary["Loan intent"] = row[5]
      request_dictionary["Loan grade"] = row[6]
      request_dictionary["Amount"] = row[7]
      request_dictionary["Interest Rate"] = row[8]
      request_dictionary["Loan percent income"] = row[9]
      request_dictionary["Historical Defaults"] = row[10]
      request_dictionary["Credit History"] = row[11]
      request_dictionary["Predicted loan_status"] = request_pred[i]
      dll.add(request_dictionary)  # Add to the doubly linked list
      i += 1  # This keeps the index of the prediction in sync with the request data

def main():
    """
    This is the main function that runs the program
    Input: None
    Returns: None
    """

    # Part 1
    # 5.1.1:
    raw_data = load_dataset('credit_risk_train.csv')
    print(f"\n\n\033[1mPart 1\033[\n5.1.1:\n")
    print(f"\tInitial number of rows: {len(raw_data)-1}")
    data_blank = remove_blank(raw_data)
    print(f"\tRemaining number of rows: {len(data_blank)-1}")

    # 5.1.2:
    print(f"\n5.1.2:\n")
    oldies = remove_age_ninety(data_blank)
    print(f"\tRemaining number of rows: {len(oldies)}")

    # 5.1.3:
    print(f"\n5.1.3:\n")
    histogram = histo_(oldies)

    # 5.1.4:
    print(f"\n5.1.4:\n")
    piechart = piechart_(oldies)

    # 5.1.5:
    print(f"\n5.1.5:\n")
    balance_(oldies)
    # Part 1

    # Part 2
    # 5.2.1:
    print(f"\n\n\033[1mPart 2\033[\n5.2.1:\n")
    raw_test_data = load_dataset('credit_risk_test.csv')
    raw_request_data = load_dataset('loan_requests.csv')
    refined_data = data_prep(oldies)
    scaled_data = scale_data(refined_data)
    print(f"\tFirst Row of Scaled Data: {scaled_data[0]}")

    # 5.2.2:
    print(f"\n5.2.2:\n")
    refined_test_data = data_prep(raw_test_data)
    scaled_test_data = scale_data(refined_test_data)
    x_train = data_features(refined_data, scaled_data)
    x_test = data_features(refined_test_data,scaled_test_data)
    y_train = data_labels(refined_data)
    y_test = data_labels(refined_test_data)
    print(f"\tFirst Row of Features: \n\t\t{x_train[0]}\n\t\t{x_test[0]}")
    print(f"\n\tFirst Row of Labels: \n\t\t{y_train[0]}\n\t\t{y_test[0]}\n")
    clf = DecisionTreeClassifier(random_state=42)  # Create the model
    clf.fit(x_train, y_train)  # Train the model with the data
    y_pred = clf.predict(x_test)
    print(f"\tTest Accuracy: {round(accuracy_score(y_test, y_pred), 2)}\n")  # Accuracy
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")  # Precision, Recall, F1-score
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")  # Confusion Matrix
    # Part 2

    # Part 3
    # 5.3.1:
    print(f"\n\n\033[1mPart 3\033[\n5.3.1:\n")
    refined_request_data = request_data_prep(raw_request_data)
    scaled_request_data = scale_data(refined_request_data)
    request_test = data_features(refined_request_data,scaled_request_data)
    print(f"\t{request_test}")
    print(f"\t{refined_request_data[0]}")
    request_pred = clf.predict(request_test, check_input=True)
    print(f"\t{request_pred}")

    # 5.3.2:
    print(f"\n5.3.2:\n")
    request_dictionary = build_dictionary(raw_request_data, request_pred)
    print(f"\t{dll.getCurrentData()}")

    # 5.3.3
    show_carousel = input(f"\n\n5.3.3:\n\tPress Enter to Proceed\n\t")
    goodbye = False
    dashes = "--------------------------------------------------"

    # The customer profile interface
    while goodbye == False:
      profile = dll.getCurrentData()
      print(f"{dashes}")
      for key, value in profile.items():

        # This is where the formatting of the output is done
        if key == "Income":
          print(f"{str(key)}: ${str(value)}\n")
        elif key == "Amount":
          print(f"{str(key)}: ${str(value)}\n")
        elif key == "Credit History":
          print(f"{str(key)}: {str(value)} years\n")
        elif key == "Predicted loan_status":
          if value == "1":
            print(f"{dashes}\n\n{str(key)}: Will default\nRecommend: Reject\n\n{dashes}")
          elif value == "0":
            print(f"{dashes}\n\n{str(key)}: Will not default\nRecommend: Accept\n\n{dashes}")
        elif key == "Historical Defaults":
          if value == "Y":
            print(f"{str(key)}: Yes\n")
          elif value == "N":
            print(f"{str(key)}: No\n")
        else:
          print(f"{str(key)}: {str(value)}\n")

      # Profile navigation 
      print(f'\nInput "1" to go next, "2" to go back, and "0" to quit')
      invalid = True
      while invalid == True:
        shall_we = input()
        if shall_we == "1":  # Next
          dll.moveNext()
          invalid = False
        elif shall_we == "2":  # Previous
          dll.movePrevious()
          invalid = False
        elif shall_we == "0":  # Quit
          goodbye = True
          invalid = False
        else:  # Invalid input
          print('Invalid Input\nInput "1" to go next, "2" to go back, and "0" to quit')

    print("\nGoodbye...")
    # Part 3

# Global variable for the doubly linked list
dll = Carousel()

if __name__ == "__main__":
    main()