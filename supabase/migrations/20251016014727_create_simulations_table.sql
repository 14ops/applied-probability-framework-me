/*
  # Create simulations table for Mines Strategy Framework

  1. New Tables
    - `simulations`
      - `id` (uuid, primary key)
      - `strategy` (text) - Strategy name (takeshi, lelouch, kazuya, senku)
      - `board_size` (integer) - Size of the game board
      - `mine_count` (integer) - Number of mines on the board
      - `initial_bankroll` (numeric) - Starting bankroll amount
      - `final_bankroll` (numeric, nullable) - Ending bankroll amount
      - `total_rounds` (integer, default 0) - Number of rounds completed
      - `status` (text, default 'pending') - Simulation status (pending, running, completed, failed)
      - `created_at` (timestamptz) - Timestamp of creation
      - `updated_at` (timestamptz) - Timestamp of last update
  
  2. Security
    - Enable RLS on `simulations` table
    - Add policy for anyone to insert simulations (public demo)
    - Add policy for anyone to read simulations
    - Add policy for anyone to update simulations
*/

CREATE TABLE IF NOT EXISTS simulations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy text NOT NULL,
  board_size integer NOT NULL DEFAULT 5,
  mine_count integer NOT NULL DEFAULT 3,
  initial_bankroll numeric NOT NULL DEFAULT 1000,
  final_bankroll numeric,
  total_rounds integer DEFAULT 0,
  status text DEFAULT 'pending',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public insert on simulations"
  ON simulations
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public select on simulations"
  ON simulations
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public update on simulations"
  ON simulations
  FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_simulations_created_at ON simulations(created_at DESC);
