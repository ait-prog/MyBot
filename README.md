# AI-Driven Discord Chatbot

![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

An advanced AI-powered Discord bot with natural language processing capabilities, economy system, private rooms, and role management.

## Features

### 🤖 AI Chat System
- Private AI chat rooms with customizable personalities
- Context-aware conversations with memory
- Command to clear chat history
- Personality customization

### 💰 Economy System
- Currency with coins and diamonds
- Daily rewards system
- Work commands to earn currency
- Trading and dueling between users
- Robbery mechanics (with risks!)

### 🏠 Private Rooms
- Create custom voice/text channels
- Invite system for rooms
- Room management commands
- Role-based permissions

### 💍 Social Features
- Marriage system between users
- Romantic interactions (kiss, hug)
- Divorce mechanics

### 🎭 Role Management
- Custom role creation
- Role color/name customization
- Role transfer system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-discord-bot.git
cd ai-discord-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Discord token:
```env
DISCORD_TOKEN=your_bot_token_here
```

4. Run the bot:
```bash
python bot.py
```

## Configuration

Edit `config.py` to customize:
- Currency names and emojis
- Prices for various features
- Channel IDs for special functions
- Timeout settings

## Command Reference

### AI Room Commands
- `!airoom` - Create a private AI chat room (costs 5000 coins)
- `!personality [text]` - Change AI personality
- `!memory` - Show chat memory status
- `!clear` - Clear chat history

### Economy Commands
- `!balance` - Check your balance
- `!daily` - Claim daily reward
- `!work` - Earn coins (1h cooldown)
- `!give @user amount` - Transfer coins
- `!duel @user amount` - Challenge to a duel

### Room Commands
- `!createroom [name]` - Create private room (2000 coins)
- `!invite @user` - Invite to your room
- `!kick @user` - Remove from your room
- `!rename [new name]` - Rename your room

### Marriage Commands
- `!propose @user` - Propose marriage
- `!divorce` - End your marriage
- `!marriage` - Show marriage info
- `!kiss @user` - Kiss your spouse
- `!hug @user` - Hug your spouse

### Role Commands
- `!buyrole [name]` - Buy a custom role (1000 coins)
- `!rolecolor [role] [hex color]` - Change role color
- `!rolename [role] [new name]` - Rename role
- `!transferrole [role] @user` - Transfer role ownership

## Technical Architecture

graph TD
    
    A[Discord Bot] --> B[AI Chat Module]
    A --> C[Economy System]
    A --> D[Room Management]
    A --> E[Role System]
    A --> F[Marriage System]
    
    B --> G[Neural Network]
    B --> H[Chat History]
    
    C --> I[Database]
    D --> I
    E --> I
    F --> I
    
    G --> J[Model Training]
    G --> K[Response Generation]


## Dependencies

- Python 3.8+
- discord.py
- TensorFlow/Keras
- aiosqlite
- python-dotenv

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements.

## License

MIT License
