// Code generated by stanc v2.33.1
#include <stan/model/model_header.hpp>
namespace irt_2pl_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 26> locations_array__ =
  {" (found before start of program)",
  " (in 'irt_2pl.stan', line 7, column 2 to column 28)",
  " (in 'irt_2pl.stan', line 8, column 2 to column 18)",
  " (in 'irt_2pl.stan', line 10, column 2 to column 24)",
  " (in 'irt_2pl.stan', line 11, column 2 to column 23)",
  " (in 'irt_2pl.stan', line 13, column 2 to column 12)",
  " (in 'irt_2pl.stan', line 14, column 2 to column 24)",
  " (in 'irt_2pl.stan', line 15, column 2 to column 14)",
  " (in 'irt_2pl.stan', line 18, column 2 to column 29)",
  " (in 'irt_2pl.stan', line 19, column 2 to column 33)",
  " (in 'irt_2pl.stan', line 21, column 2 to column 25)",
  " (in 'irt_2pl.stan', line 22, column 2 to column 28)",
  " (in 'irt_2pl.stan', line 24, column 2 to column 22)",
  " (in 'irt_2pl.stan', line 25, column 2 to column 25)",
  " (in 'irt_2pl.stan', line 26, column 2 to column 28)",
  " (in 'irt_2pl.stan', line 29, column 4 to column 50)",
  " (in 'irt_2pl.stan', line 28, column 19 to line 30, column 3)",
  " (in 'irt_2pl.stan', line 28, column 2 to line 30, column 3)",
  " (in 'irt_2pl.stan', line 2, column 2 to column 17)",
  " (in 'irt_2pl.stan', line 3, column 2 to column 17)",
  " (in 'irt_2pl.stan', line 4, column 8 to column 9)",
  " (in 'irt_2pl.stan', line 4, column 11 to column 12)",
  " (in 'irt_2pl.stan', line 4, column 2 to column 38)",
  " (in 'irt_2pl.stan', line 8, column 9 to column 10)",
  " (in 'irt_2pl.stan', line 11, column 18 to column 19)",
  " (in 'irt_2pl.stan', line 15, column 9 to column 10)"};
class irt_2pl_model final : public model_base_crtp<irt_2pl_model> {
 private:
  int I;
  int J;
  std::vector<std::vector<int>> y;
 public:
  ~irt_2pl_model() {}
  irt_2pl_model(stan::io::var_context& context__, unsigned int
                random_seed__ = 0, std::ostream* pstream__ = nullptr)
      : model_base_crtp(0) {
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "irt_2pl_model_namespace::irt_2pl_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 18;
      context__.validate_dims("data initialization", "I", "int",
        std::vector<size_t>{});
      I = std::numeric_limits<int>::min();
      current_statement__ = 18;
      I = context__.vals_i("I")[(1 - 1)];
      current_statement__ = 18;
      stan::math::check_greater_or_equal(function__, "I", I, 0);
      current_statement__ = 19;
      context__.validate_dims("data initialization", "J", "int",
        std::vector<size_t>{});
      J = std::numeric_limits<int>::min();
      current_statement__ = 19;
      J = context__.vals_i("J")[(1 - 1)];
      current_statement__ = 19;
      stan::math::check_greater_or_equal(function__, "J", J, 0);
      current_statement__ = 20;
      stan::math::validate_non_negative_index("y", "I", I);
      current_statement__ = 21;
      stan::math::validate_non_negative_index("y", "J", J);
      current_statement__ = 22;
      context__.validate_dims("data initialization", "y", "int",
        std::vector<size_t>{static_cast<size_t>(I), static_cast<size_t>(J)});
      y = std::vector<std::vector<int>>(I,
            std::vector<int>(J, std::numeric_limits<int>::min()));
      {
        std::vector<int> y_flat__;
        current_statement__ = 22;
        y_flat__ = context__.vals_i("y");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= J; ++sym1__) {
          for (int sym2__ = 1; sym2__ <= I; ++sym2__) {
            stan::model::assign(y, y_flat__[(pos__ - 1)],
              "assigning variable y", stan::model::index_uni(sym2__),
              stan::model::index_uni(sym1__));
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 22;
      stan::math::check_greater_or_equal(function__, "y", y, 0);
      current_statement__ = 22;
      stan::math::check_less_or_equal(function__, "y", y, 1);
      current_statement__ = 23;
      stan::math::validate_non_negative_index("theta", "J", J);
      current_statement__ = 24;
      stan::math::validate_non_negative_index("a", "I", I);
      current_statement__ = 25;
      stan::math::validate_non_negative_index("b", "I", I);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + J + 1 + I + 1 + 1 + I;
  }
  inline std::string model_name() const final {
    return "irt_2pl_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.33.1",
             "stancflags = --filename-in-msg=irt_2pl.stan"};
  }
  // Base log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_not_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "irt_2pl_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ sigma_theta = DUMMY_VAR__;
      current_statement__ = 1;
      sigma_theta = in__.template read_constrain_lb<local_scalar_t__,
                      jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> theta =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(J, DUMMY_VAR__);
      current_statement__ = 2;
      theta = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(J);
      local_scalar_t__ sigma_a = DUMMY_VAR__;
      current_statement__ = 3;
      sigma_a = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> a =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 4;
      a = in__.template read_constrain_lb<
            Eigen::Matrix<local_scalar_t__,-1,1>, jacobian__>(0, lp__, I);
      local_scalar_t__ mu_b = DUMMY_VAR__;
      current_statement__ = 5;
      mu_b = in__.template read<local_scalar_t__>();
      local_scalar_t__ sigma_b = DUMMY_VAR__;
      current_statement__ = 6;
      sigma_b = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> b =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 7;
      b = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(I);
      {
        current_statement__ = 8;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_theta, 0, 2));
        current_statement__ = 9;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(theta, 0,
                         sigma_theta));
        current_statement__ = 10;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_a, 0, 2));
        current_statement__ = 11;
        lp_accum__.add(stan::math::lognormal_lpdf<propto__>(a, 0, sigma_a));
        current_statement__ = 12;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(mu_b, 0, 5));
        current_statement__ = 13;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_b, 0, 2));
        current_statement__ = 14;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(b, mu_b, sigma_b));
        current_statement__ = 17;
        for (int i = 1; i <= I; ++i) {
          current_statement__ = 15;
          lp_accum__.add(stan::math::bernoulli_logit_lpmf<propto__>(
                           stan::model::rvalue(y, "y",
                             stan::model::index_uni(i)),
                           stan::math::multiply(
                             stan::model::rvalue(a, "a",
                               stan::model::index_uni(i)),
                             stan::math::subtract(theta,
                               stan::model::rvalue(b, "b",
                                 stan::model::index_uni(i))))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  // Reverse mode autodiff log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "irt_2pl_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ sigma_theta = DUMMY_VAR__;
      current_statement__ = 1;
      sigma_theta = in__.template read_constrain_lb<local_scalar_t__,
                      jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> theta =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(J, DUMMY_VAR__);
      current_statement__ = 2;
      theta = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(J);
      local_scalar_t__ sigma_a = DUMMY_VAR__;
      current_statement__ = 3;
      sigma_a = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> a =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 4;
      a = in__.template read_constrain_lb<
            Eigen::Matrix<local_scalar_t__,-1,1>, jacobian__>(0, lp__, I);
      local_scalar_t__ mu_b = DUMMY_VAR__;
      current_statement__ = 5;
      mu_b = in__.template read<local_scalar_t__>();
      local_scalar_t__ sigma_b = DUMMY_VAR__;
      current_statement__ = 6;
      sigma_b = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<local_scalar_t__,-1,1> b =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 7;
      b = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(I);
      {
        current_statement__ = 8;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_theta, 0, 2));
        current_statement__ = 9;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(theta, 0,
                         sigma_theta));
        current_statement__ = 10;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_a, 0, 2));
        current_statement__ = 11;
        lp_accum__.add(stan::math::lognormal_lpdf<propto__>(a, 0, sigma_a));
        current_statement__ = 12;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(mu_b, 0, 5));
        current_statement__ = 13;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(sigma_b, 0, 2));
        current_statement__ = 14;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(b, mu_b, sigma_b));
        current_statement__ = 17;
        for (int i = 1; i <= I; ++i) {
          current_statement__ = 15;
          lp_accum__.add(stan::math::bernoulli_logit_lpmf<propto__>(
                           stan::model::rvalue(y, "y",
                             stan::model::index_uni(i)),
                           stan::math::multiply(
                             stan::model::rvalue(a, "a",
                               stan::model::index_uni(i)),
                             stan::math::subtract(theta,
                               stan::model::rvalue(b, "b",
                                 stan::model::index_uni(i))))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    // suppress unused var warning
    (void) jacobian__;
    static constexpr const char* function__ =
      "irt_2pl_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      double sigma_theta = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      sigma_theta = in__.template read_constrain_lb<local_scalar_t__,
                      jacobian__>(0, lp__);
      Eigen::Matrix<double,-1,1> theta =
        Eigen::Matrix<double,-1,1>::Constant(J,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 2;
      theta = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(J);
      double sigma_a = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      sigma_a = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<double,-1,1> a =
        Eigen::Matrix<double,-1,1>::Constant(I,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 4;
      a = in__.template read_constrain_lb<
            Eigen::Matrix<local_scalar_t__,-1,1>, jacobian__>(0, lp__, I);
      double mu_b = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 5;
      mu_b = in__.template read<local_scalar_t__>();
      double sigma_b = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 6;
      sigma_b = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      Eigen::Matrix<double,-1,1> b =
        Eigen::Matrix<double,-1,1>::Constant(I,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 7;
      b = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(I);
      out__.write(sigma_theta);
      out__.write(theta);
      out__.write(sigma_a);
      out__.write(a);
      out__.write(mu_b);
      out__.write(sigma_b);
      out__.write(b);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      local_scalar_t__ sigma_theta = DUMMY_VAR__;
      current_statement__ = 1;
      sigma_theta = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma_theta);
      Eigen::Matrix<local_scalar_t__,-1,1> theta =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(J, DUMMY_VAR__);
      current_statement__ = 2;
      stan::model::assign(theta,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,1>>(J),
        "assigning variable theta");
      out__.write(theta);
      local_scalar_t__ sigma_a = DUMMY_VAR__;
      current_statement__ = 3;
      sigma_a = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma_a);
      Eigen::Matrix<local_scalar_t__,-1,1> a =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 4;
      stan::model::assign(a,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,1>>(I),
        "assigning variable a");
      out__.write_free_lb(0, a);
      local_scalar_t__ mu_b = DUMMY_VAR__;
      current_statement__ = 5;
      mu_b = in__.read<local_scalar_t__>();
      out__.write(mu_b);
      local_scalar_t__ sigma_b = DUMMY_VAR__;
      current_statement__ = 6;
      sigma_b = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma_b);
      Eigen::Matrix<local_scalar_t__,-1,1> b =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      current_statement__ = 7;
      stan::model::assign(b,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,1>>(I),
        "assigning variable b");
      out__.write(b);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "sigma_theta",
        "double", std::vector<size_t>{});
      current_statement__ = 2;
      context__.validate_dims("parameter initialization", "theta", "double",
        std::vector<size_t>{static_cast<size_t>(J)});
      current_statement__ = 3;
      context__.validate_dims("parameter initialization", "sigma_a",
        "double", std::vector<size_t>{});
      current_statement__ = 4;
      context__.validate_dims("parameter initialization", "a", "double",
        std::vector<size_t>{static_cast<size_t>(I)});
      current_statement__ = 5;
      context__.validate_dims("parameter initialization", "mu_b", "double",
        std::vector<size_t>{});
      current_statement__ = 6;
      context__.validate_dims("parameter initialization", "sigma_b",
        "double", std::vector<size_t>{});
      current_statement__ = 7;
      context__.validate_dims("parameter initialization", "b", "double",
        std::vector<size_t>{static_cast<size_t>(I)});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ sigma_theta = DUMMY_VAR__;
      current_statement__ = 1;
      sigma_theta = context__.vals_r("sigma_theta")[(1 - 1)];
      out__.write_free_lb(0, sigma_theta);
      Eigen::Matrix<local_scalar_t__,-1,1> theta =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(J, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> theta_flat__;
        current_statement__ = 2;
        theta_flat__ = context__.vals_r("theta");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= J; ++sym1__) {
          stan::model::assign(theta, theta_flat__[(pos__ - 1)],
            "assigning variable theta", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      out__.write(theta);
      local_scalar_t__ sigma_a = DUMMY_VAR__;
      current_statement__ = 3;
      sigma_a = context__.vals_r("sigma_a")[(1 - 1)];
      out__.write_free_lb(0, sigma_a);
      Eigen::Matrix<local_scalar_t__,-1,1> a =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> a_flat__;
        current_statement__ = 4;
        a_flat__ = context__.vals_r("a");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
          stan::model::assign(a, a_flat__[(pos__ - 1)],
            "assigning variable a", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      out__.write_free_lb(0, a);
      local_scalar_t__ mu_b = DUMMY_VAR__;
      current_statement__ = 5;
      mu_b = context__.vals_r("mu_b")[(1 - 1)];
      out__.write(mu_b);
      local_scalar_t__ sigma_b = DUMMY_VAR__;
      current_statement__ = 6;
      sigma_b = context__.vals_r("sigma_b")[(1 - 1)];
      out__.write_free_lb(0, sigma_b);
      Eigen::Matrix<local_scalar_t__,-1,1> b =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(I, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> b_flat__;
        current_statement__ = 7;
        b_flat__ = context__.vals_r("b");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
          stan::model::assign(b, b_flat__[(pos__ - 1)],
            "assigning variable b", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      out__.write(b);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"sigma_theta", "theta", "sigma_a",
                "a", "mu_b", "sigma_b", "b"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
                std::vector<size_t>{static_cast<size_t>(J)},
                std::vector<size_t>{},
                std::vector<size_t>{static_cast<size_t>(I)},
                std::vector<size_t>{}, std::vector<size_t>{},
                std::vector<size_t>{static_cast<size_t>(I)}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "sigma_theta");
    for (int sym1__ = 1; sym1__ <= J; ++sym1__) {
      param_names__.emplace_back(std::string() + "theta" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "sigma_a");
    for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
      param_names__.emplace_back(std::string() + "a" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "mu_b");
    param_names__.emplace_back(std::string() + "sigma_b");
    for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
      param_names__.emplace_back(std::string() + "b" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "sigma_theta");
    for (int sym1__ = 1; sym1__ <= J; ++sym1__) {
      param_names__.emplace_back(std::string() + "theta" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "sigma_a");
    for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
      param_names__.emplace_back(std::string() + "a" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "mu_b");
    param_names__.emplace_back(std::string() + "sigma_b");
    for (int sym1__ = 1; sym1__ <= I; ++sym1__) {
      param_names__.emplace_back(std::string() + "b" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"sigma_theta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(J) + "},\"block\":\"parameters\"},{\"name\":\"sigma_a\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"a\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(I) + "},\"block\":\"parameters\"},{\"name\":\"mu_b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"b\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(I) + "},\"block\":\"parameters\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"sigma_theta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(J) + "},\"block\":\"parameters\"},{\"name\":\"sigma_a\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"a\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(I) + "},\"block\":\"parameters\"},{\"name\":\"mu_b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"b\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(I) + "},\"block\":\"parameters\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = ((((((1 + J) + 1) + I) + 1) + 1) + I);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = ((((((1 + J) + 1) + I) + 1) + 1) + I);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = irt_2pl_model_namespace::irt_2pl_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return irt_2pl_model_namespace::profiles__;
}
#endif