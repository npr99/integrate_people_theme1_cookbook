# Integrate People Theme 1 Filename Plan

ip1 = Project mnemonic. All files start with the same mnemonic.

ip1 = Integrate People Theme 1

Example Filename: `ip1_3av1_airdata.ipynb`

Example Filename Description: Explore Air Data, first file in explore data workflow. First version of this file.

```
File Name Structure
                   s     #
                    \   /     description         extension
                     - -     /                    /
                PRJ_tsv#_xxxxxxxxxxxx_yyyy-mm-dd.ext
                   -  -                -   -  -
                  /  /                 |   |  |
                 t  v                  y   m  d


         name    length          contents
         -----------------------------------------------------------
         PRJ       3-5        Project Mnemonic (fixed string)
         _         1          padding underscore
         t         1          data science workflow task number (0-6)
         s         1          letter step within task (a,b,c..)
         v         1          v = version
         #         1          version number (1,2,3,4...)
         _         1          padding underscore
         x         5-10*      description of step

         Optional Version Control if NOT USING GitHub to manage versions
         name    length          contents
         -----------------------------------------------------------
         _         1          padding underscore
         y         4          year (2017,2018,2019,2020...)
         -         1          padding dash
         m         2          month (01,02...12)
         -         1          padding dash
         d         2          date (01,02,...31)
         
         name    length          contents
         -----------------------------------------------------------
         .         1          decimal
         ext       3          file type extension
         -----------------------------------------------------------

The data science steps include:

0 = Research Log or Project Admin
1 = Obtain Data
2 = Clean Data
3 = Explore Data
4 = Model Data
5 = Interpret Data
6 = Publish Data
```