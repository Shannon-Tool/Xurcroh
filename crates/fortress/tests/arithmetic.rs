use xurcroh_fortress::{
    execute_grid, freivalds_challenge, refresh_shares, seeded_split_matrix, split_matrix,
    split_vector, Matrix, WeightShare,
};

#[test]
fn exact_grid_reconstruction_and_refresh() {
    let weight = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
    let activation = vec![5, 7];
    let weight_shares = split_matrix(&weight, 2);
    let activation_shares = split_vector(&activation, 2);
    let shares = weight_shares
        .into_iter()
        .enumerate()
        .map(|(index, matrix)| WeightShare {
            weight_index: index,
            challenge: freivalds_challenge(&matrix, &[11, 13]).unwrap(),
            matrix,
            provider_id: format!("provider-{index}"),
            replica_provider_id: None,
        })
        .collect::<Vec<_>>();

    let result = execute_grid(&activation_shares, &shares, 1.0).unwrap();
    assert_eq!(result, vec![19, 43]);

    let refreshed = refresh_shares(&shares[0].matrix, &shares[1].matrix);
    assert_eq!(refreshed.0 + refreshed.1, weight);
}

#[test]
fn tampering_is_attributed_to_the_responsible_role() {
    let weight = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
    let activation = vec![5, 7];
    let mut shares = split_matrix(&weight, 2)
        .into_iter()
        .enumerate()
        .map(|(index, matrix)| WeightShare {
            weight_index: index,
            challenge: freivalds_challenge(&matrix, &[11, 13]).unwrap(),
            matrix,
            provider_id: format!("provider-{index}"),
            replica_provider_id: None,
        })
        .collect::<Vec<_>>();
    shares[0].matrix.data[0] += 1;

    let error = execute_grid(&split_vector(&activation, 2), &shares, 0.0).unwrap_err();
    assert!(error.to_string().contains("provider-0"));
}

#[test]
fn seeded_shares_reconstruct_exactly() {
    let weight = Matrix::new(2, 2, vec![-8, 7, 3, -2]).unwrap();
    let shares = seeded_split_matrix(&weight, 2, [9u8; 32]);
    assert_eq!(shares[0].clone() + shares[1].clone(), weight);
}
