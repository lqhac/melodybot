#!/usr/bin/env python
# coding: utf-8

# In[8]:


from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


# In[9]:


TOKEN: Final = '7020043136:AAHxSESU5yhbeTv5OuCtcWoR9qRBYEmE1XE'
BOT_USERNAME: Final = '@pgpmusicbot'


# In[10]:


## commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello, thanks for chatting with me. I am MelodyBot;)")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I am MelodyBot. Please type in your lyrics in the format below (case insensitive) so I can generate some melodies for you!\n\nHi Bot, here are my lyrics:\n<YOUR LYRICS>")


# In[11]:


def send_document(update, context):
    chat_id = update.message.chat_id
    document = open('image1.png', 'rb')
    context.bot.send_document(chat_id, document)


# In[12]:


## handle responses
def handle_responses(text: str) -> str:
    if "lyrics" in text.lower() or "lyric" in text.lower():
        return 'Wonderful writing! Here is your melody and its piano & guitar renditions :)'
    else:
        return 'Sorry, unfortunately I cannot detect your lyrics. Could you please check your format? Hint: \n\nHi Bot, here are my lyrics:\n<YOUR LYRICS>'


# In[13]:


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    '''Connect to the backend music generator (details omitted)'''

    response: str = handle_responses(text)
    print('Bot:', response)
    await update.message.reply_text(response)

    if 'wonderful' in response.lower():
        # send midi
        chat_id = update.message.chat_id
        midi = open('/home/qihao/CS6207/3_inference/Output/Melody.mid', 'rb')
        piano = open('/home/qihao/CS6207/3_inference/Output/Piano Rendition.mp3', 'rb')
        guitar = open('/home/qihao/CS6207/3_inference/Output/Guitar Rendition.mp3', 'rb')
        await context.bot.send_document(chat_id, midi)
        await context.bot.send_document(chat_id, piano)
        await context.bot.send_document(chat_id, guitar)
        midi.close()
        piano.close()
        guitar.close()
        
        


# In[14]:


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


# In[16]:


if __name__ == "__main__":
    print('starting bot')
    app = Application.builder().token(TOKEN).build()
    ## commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    ## messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    ## errors
    app.add_error_handler(error)
    
    print('polling')
    app.run_polling(poll_interval=3)


# In[ ]:




