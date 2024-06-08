from src.utils import get_center_of_bounding_box, measure_distance


class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bounding_box):
        ball_position = get_center_of_bounding_box(ball_bounding_box)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bounding_box = player["bounding_box"]

            # left foot to ball distance
            distance_left = measure_distance(
                (player_bounding_box[0], player_bounding_box[-1]), ball_position
            )

            # right foot to ball distance
            distance_right = measure_distance(
                (player_bounding_box[2], player_bounding_box[-1]), ball_position
            )

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
