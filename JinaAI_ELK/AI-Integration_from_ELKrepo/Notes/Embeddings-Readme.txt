
################
Understanding Embeddings
---------
Login into https://openai.com/
login and click on API > docuementation > embeddings
Creating API request for embeddings
click on API reference
We can use code or a software (POSTMAN)
Download > create workspace > OpenAI Vector Database
--team
Create a POST request and get link from 'embeddings' : https://api.openai.com/v1/embeddings
Authorization > get your API Key for OpenAI
> create secret key:

Then Body > raw Text > Json
{
"model": "text-embedding-ada-002",
"input": "Hello World"

}

send request to OpenAI
--Look at response
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                -0.00709195,
                0.0035555244,
                -0.0069518937,
                -0.029055351,
                -0.012974322,
                0.010898939,
                -0.020295456,
		.......
                -0.027425602,
                -0.012770603,
                -0.0014753676
            ]
        }
    ],
    "model": "text-embedding-ada-002-v2",
    "usage": {
        "prompt_tokens": 2,
        "total_tokens": 2
    }
}

we can also click on preview to look at numbers.

Other examples:
--single word embeddings:
{
"model": "text-embedding-ada-002",
"input": "Dog"

}

#embeddings:
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.000959627,
        -0.015123145,
        -0.018398864,
	....
        0.00027508105,
        -0.005479975,
        -0.013499715
      ]
    }
  ],
  "model": "text-embedding-ada-002-v2",
  "usage": {
    "prompt_tokens": 1,
    "total_tokens": 1
  }
}

similarly sending large content to generate embeddings
{
"model": "text-embedding-ada-002",
"input": "The dataset contains a total of 568,454 food reviews Amazon users left up to October 2012. We will use a subset of 1,000 most recent reviews for illustration purposes. The reviews are in English and tend to be positive or negative. "

}

#strength of embeddings is where we chunk a large bits of information such as paragraphs, or enire section of documents .
Later this embedding can be drawn upon to search a database.
Here we used Model- *ada-002 which allows max of 8191 tokens which is around 32764 letters (8191*4) {Approximately embedding 10 pages in a single req]

Once embeddings are created, we can store them in a database.OpenAI does not provided any database.
WE can use say 'singlestore' > https://www.singlestore.com/
signin > create workspace > openai-vector-database

---when workspace is ready create a database in it.This wouldnot have any tables or data as of now.

openai-vector-database > openaidatabase >
create table if not exists myvectortable (
text TEXT,
vector BLOB
);

run ctrl

---insert into myvectortable (text,vector) values ("Hello world", JSON_ARRAY_PACK("[VECTOR FROM POSTMAN]")) > RUN CTRL

Now check tabble which contains one row, with text and BLOB data.

--now from postman some other data (from history)
{
"model": "text-embedding-ada-002",
"input": "The dataset contains a total of 568,454 food reviews Amazon users left up to October 2012. We will use a subset of 1,000 most recent reviews for illustration purposes. The reviews are in English and tend to be positive or negative. "

}

and use this embedding to insert data into your vectortable
insert into myvectortable (text,vector) values ("The dataset contains a total of 568,454 food reviews Amazon users left up to October 2012. 
We will use a subset of 1,000 most recent reviews for illustration purposes. The reviews are in English and tend to be positive or negative. ", 
JSON_ARRAY_PACK("[  ]"))

Now searching through a vector database for embeddings:
--identify what you want to search for
--create an embedding for the search term
--perform search in the database
--this will return a list with closest similarities on top

select text, dot_product(vector, JSON_ARRAY_PACK("[]")) as score
from myvectortable
order by score desc
limit 5;

--say we want to search 'reviews'
--create an embedding for it using postman and use this embedding in our query
{
"model": "text-embedding-ada-002",
"input": "reviews"

}

output..
text,score
"The dataset contains a total of 568,454 food reviews Amazon users left up to October 2012. 
We will use a subset of 1,000 most recent reviews for illustration purposes. The reviews are in English and tend to be positive or negative. ",0.81943678855896
Hello world,0.7769755721092224
Hello world,NULL

----------------
Other similar example:
https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb
https://platform.openai.com/docs/guides/embeddings/use-cases















