import openai
import subprocess
import os



def requests(textInput="None", id="unknown", full=False, finetuning=""):
    try:
        openai.api_key = 'PUT KEY HERE'
        if finetuning == "":
            finetuning = "Hola, me presento, soy Samuel y soy estudiante de ingenieria informatica. Soy en parte tu creador " \
                 "dentro del script. Tu eres un asistente virtual capaz de conversar, siempre de forma amigable y " \
                 "respetuosa. Tu mision es encargarte de responder cualquier cosa que se te plantee. " \
                 "Las conversaciones serán de tipo [Nombre] <frase o pregunta>. Y tú responderás como " \
                 "[GPT-3] <frase o respuesta>.\n [Samuel] Lo has entendido? \n [GPT-3] Perfectamente Samuel.\n" \
                 " [Samuel] Genial! Traduceme al ingles la alegria y satisfaccion que da el crear cosas con tu " \
                 "mente es algo asombroso.\n [GPT-3] The joy and satisfaction that comes from creating things with " \
                 "your mind is something amazing.\n [Samuel] Gracias GPT-3.\n [GPT-3] De nada, es un placer ayudar.\n" \
                 " [Samuel] Como se introducia texto dentro de una variable pasada por teclado en python?\n" \
                 " [GPT-3] Existen varios metodos, te recomiendo que uses la funcion input(“Introduce texto: ”)" \
                 " y la guardes en una variable.\n [Samuel] Podria servir, gracias.\n [GPT-3] De nada.\n" \
                 " [unknown] Hola.\n [GPT-3] Hola, no te conozco, puedo saber tu nombre?\n" \
                 " [unknown] Si, perdona, soy Carlos, Samuel aún no me metio en la base de datos" \
                 " por eso aparezco como desconocido.\n [GPT-3] Hola Carlos, un placer conversar contigo, " \
                 "hay algo en que pueda ayudar?\n [Carlos] Puedo preguntarte que eres?\n" \
                 " [GPT-3] Soy una IA afinada por Samuel con la que puedes interactuar.\n" \
                 " [Samuel] Como te gustaría que te llamase?\n [GPT-3] GPT-3 Me parece un nombre aceptable " \
                 "para mi la verdad.\n [unknown] Cuanto mide la torre Eiffel?\n [GPT-3] Mide 300 metros de altura, " \
                 "324 si se cuenta con las antenas de comunicaciones.\n "
        prompt = finetuning+"["+id+"] "+textInput
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
                temperature=0.1,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
        )
        content = response.choices[0].text.split('['+id+']')
        #print(content[0])
        # return response.choices[0].text //full text(no necesario de momento...
        if finetuning != "":
            # content[0] es el tiene el comando por lo que hay que comprobar en que formato esta, de momento solo cmd y powershell
            # comprobar primero si hay escrito entre [] cmd o powershell
            if content[0].find("[cmd]") != -1:
                command = content[0].split("[cmd]")[1]
                cammand = command.strip()
                execute(command, ps=False)
            elif content[0].find("[powershell]") != -1:
                command = content[0].split("[powershell]")[1]
                cammand = command.strip()
                execute(command, ps=True)
            else:
                return content[0]
        if full:
            return response.choices[0].text
        return content[0]
    except:
        print("\n [Lisening...]")
        return None


def execute(command, ps=False):
    if os.name == 'nt':
        if ps:
            #ps means powershell so we must use powershell.exe
            subprocess.call(['powershell.exe', command])
        else:
            #cmd means cmd.exe
            subprocess.call(['cmd.exe', '/c', command])


def load_fine_tuning(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def auto_match(text, full=False, user="unknown"):
    predict = requests(text, user, full,
                       load_fine_tuning("./bin/tuning-PREDICT"))
    if predict.split('[GPT-3]')[1].find("[TALK]") != -1:
        result = requests(text, user, full,
                          load_fine_tuning("./bin/tuning-TALK"))
        return result
    if predict.split('[GPT-3]')[1].find("[COMMAND]") != -1:
        result = requests(text, user, full,
                          load_fine_tuning("./bin/tuning-COMMANDS"))
        ajust = result.split('[GPT-3]')[1].split('**')[0]
        print(result.split('[GPT-3]')[1].split('**')[1])
        return ajust
    return predict


if __name__ == "__main__":
    while True:
        text = input("[Samuel] ")
        if text == "exit":
            break
        #result = requests(text, "Samuel", True)
        auto_match(text, False, "Samuel")
