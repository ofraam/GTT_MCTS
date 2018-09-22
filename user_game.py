import copy


class gameInstance:
  def __init__(self, rows):
    self.actions = []
    self.solution = ""
    self.time = ""
    self.board_positions = []

    if len(rows)>0:
        self.userid = rows[0]['userid']
    practice_start = False
    for row in rows:
        if row['key'] == 'start':
            if (practice_start):
                self.actions.append((row['key'], row['value']))
            else:
                practice_start = True
        if row['key'] in ('click', 'undo', 'reset'):
            self.actions.append((row['key'], row['value']))
        elif row['key'] == 'timeSolution':
            self.time = row['value']
        elif row['key'] == 'best_move':
            self.solution = row['value']
            break;
    for act in self.actions:
        self.board_positions.append(self.board_to_matrix(act[1]))
    # print 'done'


  def get_actions(self, action_type = None):
    if action_type is None:
        return self.actions
    else:
        return self.get_actions_by_type(action_type)


  def get_actions_by_type(self, action_type):
    filtered_actions = []
    for act in self.actions:
        if act[0] == action_type:
            filtered_actions.append(act)
    return filtered_actions


  def board_to_matrix(self, board):
      positions = []
      board2 = board[1:len(board) - 1]
      b = board2.split(']')
      for row in b:
          position_row = []
          # print row
          row_new = row[1:]
          row_new_final = row_new.replace('[', "")
          if (len(row)) > 2:
              marks = row_new_final.split(',')
              for mark in marks:
                  position_row.append(int(mark))
              positions.append(copy.deepcopy(position_row))
      return positions


  def get_action_by_index(self, action_index):
      before_position = self.board_positions[action_index-1]
      after_position = self.board_positions[action_index]
      for row in range(len(before_position)):
          for col in range(len(before_position[0])):
              if before_position[row][col]!=after_position[row][col]:
                  return (row,col,after_position[row][col])
                  break
      return None
