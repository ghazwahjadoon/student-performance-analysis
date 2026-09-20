import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# 1. LOAD DATA


df = pd.read_csv("data/students.csv")




# 2. INITIAL DATA EXPLORATION

print("First 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns)


# 3. CHECK MISSING VALUES


nullValues = df.isnull().sum()

print("\nNull Values in Each Column:")
print(nullValues)



# 4. CHECK DUPLICATE ROWS


print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())



# 5. CHECK SUSPICIOUS / INVALID VALUES


print("\nInvalid Attendance Values:")
print(df[df["Attendance"] > 100])

print("\nNegative Attendance Values:")
print(df[df["Attendance"] < 0])

print("\nNegative Study Hours:")
print(df[df["Study_Hours"] < 0])

print("\nNegative Previous Scores:")
print(df[df["Previous_Score"] < 0])

print("\nInvalid Final Scores:")
print(df[df["Final_Score"] > 100])

print("\nNegative Final Scores:")
print(df[df["Final_Score"] < 0])

print("\nNegative Sleep Hours:")
print(df[df["Sleep_Hours"] < 0])



# 6. HANDLE MISSING VALUES


df["Study_Hours"] = df["Study_Hours"].fillna(
    df["Study_Hours"].median()
)

df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].median()
)

df["Previous_Score"] = df["Previous_Score"].fillna(
    df["Previous_Score"].median()
)

df["Sleep_Hours"] = df["Sleep_Hours"].fillna(
    df["Sleep_Hours"].median()
)

df["Final_Score"] = df["Final_Score"].fillna(
    df["Final_Score"].median()
)



# 7. REMOVE DUPLICATES


df.drop_duplicates(inplace=True)



# 8. REMOVE INVALID VALUES


df.drop(
    df[df["Attendance"] > 100].index,
    inplace=True
)

df.drop(
    df[df["Final_Score"] > 100].index,
    inplace=True
)



# 9. VERIFY DATA CLEANING


print("\n========== AFTER CLEANING ==========")

print("Duplicate Rows:")
print(df.duplicated().sum())

print("\nInvalid Attendance:")
print((df["Attendance"] > 100).sum())

print("\nInvalid Final Scores:")
print((df["Final_Score"] > 100).sum())

print("\nMissing Values:")
print(df.isnull().sum())


# 10. BASIC DATA ANALYSIS


print("\n========== BASIC ANALYSIS ==========")

print("\nAverage Final Score:")
print(df["Final_Score"].mean())

print("\nHighest Final Score:")
print(df["Final_Score"].max())

print("\nLowest Final Score:")
print(df["Final_Score"].min())

print("\nAverage Study Hours:")
print(df["Study_Hours"].mean())

print("\nMaximum Study Hours:")
print(df["Study_Hours"].max())

print("\nMinimum Study Hours:")
print(df["Study_Hours"].min())



# 11. GROUP ANALYSIS




print("\nAverage Final Score by Gender:")
print(
    df.groupby("Gender")["Final_Score"].mean()
)

print("\nAverage Study Hours by Gender:")
print(
    df.groupby("Gender")["Study_Hours"].mean()
)

print("\nAverage Attendance by Gender:")
print(
    df.groupby("Gender")["Attendance"].mean()
)



# 12. TOP STUDENTS

top_students = df.sort_values(
    "Final_Score",
    ascending=False
)


print(top_students.head())






print(
    df[["Study_Hours", "Final_Score"]].corr()
)


correlation = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Final_Score",
        "Sleep_Hours"
    ]
].corr()


print(correlation)


# 14. HISTOGRAM


plt.hist(df["Final_Score"])

plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Scores")

plt.show()


# 15. SCATTER PLOT


plt.scatter(
    df["Study_Hours"],
    df["Final_Score"]
)

plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")

plt.show()


# 16. AVERAGE FINAL SCORE BY GENDER


gender_scores = df.groupby("Gender")["Final_Score"].mean()

gender_scores.plot(kind="bar")

plt.xlabel("Gender")
plt.ylabel("Average Final Score")
plt.title("Average Final Score by Gender")

plt.show()


# 17. AVERAGE STUDY HOURS BY GENDER


gender_study = df.groupby("Gender")["Study_Hours"].mean()

gender_study.plot(kind="bar")

plt.xlabel("Gender")
plt.ylabel("Average Study Hours")
plt.title("Average Study Hours by Gender")

plt.show()


# 18. BOX PLOT


df.boxplot(
    column="Final_Score",
    by="Gender"
)

plt.title("Final Score Distribution by Gender")
plt.suptitle("")

plt.xlabel("Gender")
plt.ylabel("Final Score")

plt.show()






# PROJECT COMPLETE

print("Student Performance Analysis completed successfully.")