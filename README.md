# Association Rule Algorithms in Machine Learning (Apriori)

Machine Learning is all about finding patterns in the past data and using it to take future decisions. There are many types of algorithms available in machine learning like classification, regression, clustering and etc. Based on our requirement and resources we will choose one. Association rules are another type of machine learning algorithms which will allow us to find the relationships between two or more elements.

In association rules we will try to find data item that have association with another data item, to explain this let's take the super market for example, In super market arranging the products take lots of decision making, because they need to decide how to make the customer buy more products. Let's say if you keep a complementing product close to another product such as tea and sugar then there is a high possibility where customer will buy both.

Finding association between tea and sugar is quite obvious, but do you know baby pampers and the beer have an association between them. Did you ever think a baby pamper and beer would have association between each other. This is the story, In USA it's common that daddy's take care of their babies in the weekend, so while taking care of their babies they were chilling out with a beer.

OK, are you already tired of thinking about how to find such hidden associations, then don't worry association rules are there to rescue. In this example I am going to use a type of association algorithm call **apriori** to find buying patterns on a retail store's daily transactions for one month.

In this article I will paste the code first and explain it afterwards.

    import pandas as pd
    dataset = pd.read_csv("data.csv", header=None)
    
    # apriori expects the input as list of list
    # [ [item1, item2], [item2, item3, item4], ..... ] ]
    transactions = []
    for i in range(0, len(dataset)):
        transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
      
    from apyori import apriori
    associations = apriori(transactions, min_support = 0.004, min_confidence = 0.4, min_lift = 3, min_length = 2)
    
    print(list(associations))

The first couple of lines are just importing the dataset and converting it into a pandas data frame so that we can access the data easily.

The next step is very important, apriori algorithm takes the input as list of lists, so we need to make our dataset into a list of list format, the nested loop will do the job for us. After these lines the transaction will have list of transactions, each transaction will have the items that were purchased.

Then comes the important part, using the apriori algorithm, in-order to import the apriori algorithm you need to download the [apyori file](https://github.com/ymoch/apyori/blob/master/apyori.py) and put it in the same directory where you have your file. Now let's go to the most important part, which is understanding the apriori algorithm. As you can see in the above code, we have passed transactions along four parameters which are support, confident and lift. Let's see what each word means.

**Support** : The frequency of an item being purchased in all the transaction. In simple terms how many times a product being purchased. Let's see how to choose a value for the support parameter. This dataset has 7500 transaction. So I thought to consider items that were in at least 30 transactions. So the support should be **30/7500 = 0.004**.


> Support (Product A) = No. of. Transaction A / No. of. Total Transaction


**Confident** : The frequency of purchasing a product along with another product. In simple terms how many times we bought products together divided by number of time a product is purchased, let's say we have bought "Product A" 100 time, and among those 100 transaction we bought "Product B" 60 times, then the confident (buying "Product B" if a customer buy "Product A") is 0.6. Here we set 0.4 as the minimum confident.

> Confidence (Product A -> Product B) = No. of. Transaction (A->B) / No. of. Transaction (A)

**Lift :** Lift is the boost we get from suggesting a product to a person who purchased some other product and to a normal person who didn't buy the other product. Let's take an example, let's say people have bought "Product B" 30 times, and People have bought "Product A" 500 times. Also 25 of the 30 "Product B" transactions have "Product A" as well. Therefore a normal person buying "Product B" percentage (Support B) is **30 / 7500 = 0.004** and a person buying "Product B" who bought "Product A" is **25 / 500 = 0.05**. So the boost we obtain is **0.05 / 0.004 = 12.5%**. This is called the Lift.

> Lift (Product A -> Product B) = Confidence (A -> B) / Support (B)

**Min_length** : This indicate what is the minimum number of items that should be in a transaction.

That's all that we have to know about apriori, then the algorithm will go through the dataset and find associations which match our Support, Confident, Lift Value and produce a result set. Then we will convert it into a python list and visualize the different associations. Let's visualize one such resulting relationships.The below result is the first record in the result list that get printed.

> RelationRecord(items=frozenset({'spaghetti', 'ground beef', 'cooking oil'}), 
> support = 0.004799360085321957,
> ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef', 'cooking oil'}), 
> items_add=frozenset({'spaghetti'}),
> confidence=0.5714285714285714, lift=3.2819951870487856)])

As you can see above the chosen item collection is **{'spaghetti', 'ground beef', 'cooking oil'}** and the support for this is 0.0047. The association rule is if someone buys 'ground beef' and 'cooking oil' (look at item_base) then they would buy 'spaghetti' (look at items_add). The confident for the above association is 0.57 and lift is 3.2.

That's it.