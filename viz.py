import arcade
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Typing Animation Example"

class TypingAnimationApp(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        
        # Application state variables
        self.background_color = arcade.color.AMAZON
        self.current_text = ""
        
        # Animation variables
        self.circle_x = SCREEN_WIDTH // 2
        self.circle_y = SCREEN_HEIGHT // 2
        self.circle_radius = 0
        self.circle_color = arcade.color.WHITE

    def on_draw(self):
        """ Render the screen. """
        self.clear()
        
        # Draw the animated circle if it has a radius
        if self.circle_radius > 0:
            arcade.draw_circle_filled(self.circle_x, self.circle_y, self.circle_radius, self.circle_color)
            
        # Draw the text the user has typed
        arcade.draw_text(
            f"Typed: {self.current_text}",
            start_x=50,
            start_y=SCREEN_HEIGHT - 100,
            color=arcade.color.WHITE,
            font_size=24,
            bold=True
        )

    def on_update(self, delta_time):
        """ Movement and animation logic (runs ~60 times a second) """
        # Make the ripple effect expand and fade out
        if self.circle_radius > 0:
            self.circle_radius += 5
            
            # Reset after it gets too big
            if self.circle_radius > 300:
                self.circle_radius = 0

    def on_key_press(self, key, modifiers):
        """ Triggered instantly every time a key is pressed """
        
        # Handle Backspace
        if key == arcade.key.BACKSPACE:
            self.current_text = self.current_text[:-1]
            return
            
        # Convert the key code to an actual character
        char = chr(key) if 32 <= key <= 126 else ""
        
        if char:
            self.current_text += char
            
            # --- ANIMATION TRIGGER LOGIC ---
            # 1. Randomize the background color on every keystroke
            self.background_color = (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))
            
            # 2. Trigger a "ripple" effect at a random spot
            self.circle_x = random.randint(100, SCREEN_WIDTH - 100)
            self.circle_y = random.randint(100, SCREEN_HEIGHT - 100)
            self.circle_radius = 10  # Starts the animation in on_update
            self.circle_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))

def main():
    app = TypingAnimationApp()
    arcade.run()

if __name__ == "__main__":
    main()