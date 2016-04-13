# CSCI544-Team-7
Comment Summarization System for a Social Media Post

Introduction

Social media and streams, such as Twitter, Facebook or YouTube, contain user comments that depict valuable user opinions. On initial viewing these comments may seem less than useful, full of replication, extreme views, petty arguments and spam, but when studied closely and analyzed effectively they provide multiple view points and a wide range of experience and knowledge from many different sources. Summarizing the content of these comments allows users to interact with the data at a higher level as it gives an overview impression of the conversation that has occurred. For any particular informative web resource, it is challenging to quickly ascertain the overall themes and thrusts of the mass of user-contributed comments. Several approaches exist for selectively focusing attention, such as editorial selection, collaborative recommendation, keyword cloud. While these and related methods provide a first-step towards comprehending the large amount of user-contributed comments in social media, the overall goal of this project is to develop the algorithms and methods necessary for ongoing information dissemination from such a large and growing body of content.

Method

Materials: SenTube—a dataset of user generated comments on YouTube videos annotated for information content and sentiment polarity. It contains annotations that allow to develop classifiers for several important NLP tasks: (i) sentiment analysis, (ii) text categorization, (iii) spam detection, and (iv) prediction of comment informativeness. We rely on Italian corpus that contains usercomments for about 200 YouTube videos to achieve this task.

Procedure: We propose to automatically summarize user-contributed comments through a process of identifying and extracting key informative comments. Our overall approach is to (i) identify groups of thematically-related comments through an application of topic-based clustering based on Latent Dirichlet Allocation, (ii) then rank comments in order to identify important and informative comments within each cluster using precedence based ranking.

Evaluation: Our evaluation is designed with an aim to understand if the model does indeed capture salient features for predicting community preference. As a baseline, we can measure the effectiveness of the learned model by comparing the predicted rank order of the comments to the ground truth rank order, annotated in the SenTube Corpus. We then evaluate the quality of the predictions using the well-known NDCG measure for evaluatingthe quality of top-k lists and then compare against the NDCG of traditional document

summarization approaches: MEAD and LexRank.