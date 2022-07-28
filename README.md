# United Nations General Assembly NLP Project
*Fundamentals of Data Science Course (5294FUDS6Y)*

![ScreenShot](https://raw.github.com/jah377/NLP_GeneralAssembly/main/report/github_wordcloud.png)

This repository contains code corresponding to the analyses presented in the final paper for the Fundamentals of Data Science course, in partial fulfillment for the degree of MSc. in Data Science at the University of Amsterdam.

## Executive Summary

The United Nations General Assembly offers a unique context to study international politics, as the speeches represent its member states official perception of and stance on current political developments. In this paper, we explore if differences in development strata, based on the Human Development Index (HDI), are reflected in the General Assembly speeches. Exploratory data analyses aimed to identify differences in word usage and frequency, speech sentiment, class-to-class reference patterns and investigate the feasibility of a multinomial classification algorithm. Our results show minimal differences in word usage between the development classes, with cosine similarities ranging from (0.93-0.99). Moreover, we find no relationship between HDI score and speech sentiment and that the proportion of references to countries in the four classes are similar across all groups. Finally, we implemented a logistic regression model with an accuracy of 71%, thus indicating some correlation between HDI class and word usage in speeches. However, the top 5 words are very general and don’t relate to overarching topics or themes. *In toto*, we did not find evidence that there is no relationship between the level of development and the content of the corresponding speeches during the General Assembly.

The submitted manuscript can be found within the repo [link](https://raw.github.com/jah377/NLP_GeneralAssembly/main/report/manuscript.pdf)