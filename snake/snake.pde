import java.util.Deque;
import java.util.LinkedList;
import java.lang.Math;
import java.util.ArrayList;
PVector background = new PVector(0,0,255);
class Snake{
    PVector position;
    int length;
    boolean dead;
    int speed;
    Heading direction;
    int size_of_grid;
    Deque<Square> queue;
    Grid grid;
    boolean growing;
    PVector colour = new PVector(0,255, 0);
    
    Snake(int s, Grid grid){
     position = new PVector(1,1);
     length = 1;
     dead = false;
     speed = 1;
     direction = Heading.SOUTH;
     size_of_grid = s;
     queue = new LinkedList();
     this.grid = grid;
     growing = false;
    }
    
    void move(){
     switch(direction){
      case SOUTH:
        moveDown();
        break;
      case NORTH:
        moveUp();
        break;
      case EAST:
        moveRight();
        break;
      case WEST:
        moveLeft();
        break;
     }
    }
    void moveRight(){
      if(length == 1){
        direction = Heading.EAST;
        if(position.x != size_of_grid - 1){
          //position.x++;          
        } else{
          dead = true; 
        }
      } else{
         if(direction != Heading.WEST){
           direction = Heading.EAST;
            if(position.x != size_of_grid - 1){
            //position.x++;          
          } else{
            dead = true; 
          }
        }
      }
    }
    
    void moveLeft(){
      if(length == 1){
        direction = Heading.WEST;
        if(position.x != 0){
          //position.x--;
        } else{
          dead = true; 
        }
      } else{
         if(direction != Heading.EAST){
           direction = Heading.WEST;
            if(position.x != 0){
            //position.x--;    
          } else{
            dead = true; 
          }
        }
      }
    }
    
    void moveUp(){
      if(length == 1){
        direction = Heading.NORTH;
        if(position.y != 0){
          //position.y--;
        } else{
          dead = true; 
        }
      }else{
         if(direction != Heading.SOUTH){
           direction = Heading.NORTH;
            if(position.y!= 0){
            //position.y--;          
          } else{
            dead = true; 
          }
        }
      }
    }
    
    void moveDown(){
      if(length == 1){
        direction = Heading.SOUTH;
        if(position.y != size_of_grid - 1){
          //position.y++;
        } else{
          dead = true; 
        }
      } else{
         if(direction != Heading.NORTH){
           direction = Heading.SOUTH;
            if(position.y != size_of_grid - 1){
            //position.y++;          
          } else{
            dead = true; 
          }
        }
      }
    }
    
    void addToQueue(Square square){
        
       if(queue.peek() != null){
          Square old_square =  queue.removeFirst();
          old_square.colour = background;
        }
        
        queue.addLast(square);
        
    }
    
    void addToQueueTarget(Square square){
       queue.addLast(square);
       length++;
    }
    
  
      
    
}

class Square{
   int side_length;
   int[] id_number;
   PVector position;
   PVector colour;
   
   Square(int h, int[] id){
    side_length = h; 
    id_number = id;
    position = new PVector(id[0] * side_length, id[1] * side_length);
    colour = background;
   }
   
    void draw_square(PVector colour){
      fill(colour.x, colour.y, colour.z);
      stroke(colour.x, colour.y, colour.z);
      //stroke(background.x, background.y, background.z);

      square(position.x, position.y, side_length);
    }
   
}

class Grid{
   Square[][] grid;
   ArrayList<PVector> v_lines; // (x1, y1, y2)
   ArrayList<PVector> h_lines; // (y1, x1, x2)
   int grid_length;
   int grid_height;
   int square_size;
   Square target;
   
   
   Grid(int size, int window_size){
    grid = new Square[size][size];
    v_lines = new ArrayList();
    grid_length = size;
    grid_height = size;
    square_size = window_size/grid_height;
    for(int i = 0; i < grid_length; i++){
      for(int j = 0; j < grid_height; j++){
        grid[i][j] = new Square(square_size, new int[]{i, j});
        v_lines.add(new PVector(i*square_size, 0, window_size));
        v_lines.add(new PVector(j*square_size, 0, window_size));
       }
     }
     ArrayList<PVector> params = new ArrayList();
     params.add(new PVector(0,0));
     target = generateTarget(params);
   }
   
   void show(){
     
     
     
     for(int i = 0; i < grid_length; i++){
      for(int j = 0; j < grid_height; j++){
        Square current_square = grid[i][j]; 
        current_square.draw_square(current_square.colour);
       }
     }
     
   }
   
   void set_colour(PVector id, PVector colour){
      grid[(int)id.x][(int)id.y].colour = colour; 
   }
   
   Square generateTarget(ArrayList<PVector> p){
     int xpos = (int) random(0,grid_length);
     int ypos = (int) random(0,grid_length);
     for(PVector pos: p){
       while(xpos == pos.y && ypos == pos.x){
          xpos = (int) random(0,grid_length);
          ypos = (int) random(0,grid_length);
       } 
     }
     
     
     
  
     
     Square targ = grid[ypos][xpos];
     targ.colour = new PVector(255, 0, 0);
     return targ;
     
   }
}

Grid g;
Snake s;
int grid_size = 30;
PVector target_pos;
int count = 0;
int seconds = 0;
boolean start = false;
void setup(){
  size(700,700);
  background(0);
   g = new Grid(grid_size, height);
   s = new Snake(grid_size, g);
   s.addToQueue(g.grid[(int)s.position.x][( int)s.position.y]);
   target_pos = new PVector(g.target.id_number[0], g.target.id_number[1]);
}   

void draw(){
   //square(1,1,50);
  if(key == 's'){
     start = true; 
  }
  if(start){
    if(!s.dead){
    for(Square block: s.queue){
      block.colour =  s.colour;
    }
    //g.set_colour(s.position, new PVector(0,255,0));
    if(seconds % 5 == 0){
      switch(s.direction){
       case SOUTH:
         if(s.position.y < s.size_of_grid - 1)
           s.position.y++;
          else{
            die();
          }
         break;
       case NORTH:
         if(s.position.y > 0)
             s.position.y--;
         else{
           die();
         }
         break;
       case WEST:
         if(s.position.x > 0)
           s.position.x--;
          else{
            die();
          }
          break;
       case EAST:
         if(s.position.x < s.size_of_grid - 1)
           s.position.x++;
          else{
             die();
          }
         break;
      }
     
     if(!s.dead){
       selfCollision();
       ifHit();
      }
    }
    print(seconds + "\n");
    g.show();
    seconds++;
    }
  } else{
     if(key == 'r'){
       g = new Grid(grid_size, height);
       s = new Snake(grid_size, g);
       s.addToQueue(g.grid[(int)s.position.x][( int)s.position.y]);
       target_pos = new PVector(g.target.id_number[0], g.target.id_number[1]);
       start = true;
     }
  }
}
  
  

void keyPressed(){
  if(keyCode == DOWN){
    s.moveDown();
  } else if(keyCode == UP){
    s.moveUp();
  } else if(keyCode ==  RIGHT){
    s.moveRight();
  } else if(keyCode == LEFT){
    s.moveLeft();
  }
  
  

   
   //print(g.target.id_number[0] +", " + g.target.id_number[1] + "\n");
   //print(s.queue.size() +","+  s.length+ "\n");
}

void ifHit(){
    if(!s.growing){
    s.addToQueue(g.grid[(int)s.position.x][(int)s.position.y]);
  } else{
    s.addToQueueTarget(g.grid[(int)s.position.x][( int)s.position.y]); 
    count++;
  }
  
  Square current_square = s.queue.peekLast();
  ArrayList<PVector> uneligible_positions = new ArrayList();
  if(current_square.id_number[0] == g.target.id_number[0] && current_square.id_number[1] == g.target.id_number[1]){
        s.growing = true;
        for(Square block: s.queue){
          uneligible_positions.add(new PVector(block.id_number[0], block.id_number[1]));
        }
        g.target = g.generateTarget(uneligible_positions);
     }
  if(count == 3){
    s.growing = false; 
    count = 0;
   }
}

void die(){
  s.dead = true;
  start = false;
  for(Square block: s.queue){
    block.colour =  new PVector(128, 128, 128);
  }
}


void selfCollision(){
   Square head = s.queue.removeLast();
   for(Square block : s.queue){
     if(head.id_number[0] == block.id_number[0] && head.id_number[1] == block.id_number[1]){
        die(); 
     }
   }
   s.queue.addLast(head);
   
}
