# Workshop Rainfall 24.04.2024

Dette repoet inneholder workshopmaterialet laget av Noria AS for Rainfall 24.04.2024. 

Har du spørsmål ta kontakt med oss:  

Ramin Hasibi - 90 64 93 91  
Martin Iden - 93 48 05 51  
Karl Arthur Unstad - 41 52 20 04


## Opprett API-nøkkel for OpenAI

1. Gå til https://openai.com/blog/openai-api

2. Lag en bruker eller log inn. Velg deretter 'API'.

3. Gå til 'Settings' som er et tannhjul på venstre side. Velg deretter 'Billing' i undermenyen.

4. Fyll på ønsket saldo. 5-10 dollar burde holde lenge.

5. Gå til 'API keys' i menyen til venstre. Ikonet er en hengelås. 

6. Opprett din egen API-nøkkel. NB: Nøkkelen blir vist bare en gang, så kopier den og lagre den et sted.

7. Nøkkelen kan brukes ved å sette: 
```python
import openai
openai.api_key = "YOUR_API_KEY"
```


